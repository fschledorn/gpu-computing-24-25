#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>
#include <hdf5.h>
#include <cuda_runtime.h>

const static int NUM_STREAMS = 8; // Number of CUDA streams
const static int CHUNK_SIZE = 1024; // Number of particles per chunk (adjust based on available memory)
const static int DEFAULT_NUM_ELEMENTS = 1024;
const static int DEFAULT_NUM_ITERATIONS = 100;
const static int DEFAULT_BLOCK_DIM = 32;

// Simulation parameters
const static float TIMESTEP = 1e-6;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)
const static float SMOOTHING = 1e-3;  

// Struct of Arrays (SoA) for particle data
struct Body_t {
    float *x, *y, *z, *w;  // Positions and mass
    float *vx, *vy, *vz;   // Velocitie
};

void printHelp(char *argv);
void printElement(Body_t *particles, int elementId, int iteration);

//
// Calculate the forces between two bodies
//
__device__ void bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
    float3 direction = make_float3(bodyA.x - bodyB.x, bodyA.y - bodyB.y, bodyA.z - bodyB.z);
    float distance = rsqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    float magnitude = - GAMMA * bodyA.w * bodyB.w / ((distance * distance * distance) + SMOOTHING);

	// Update force
    force.x += magnitude * direction.x;
    force.y += magnitude * direction.y;
    force.z += magnitude * direction.z;
}

// Kernel function for particle computation
__global__ void sharedNbody_Kernel(int numElements, Body_t body) {
    extern __shared__ float4 sharedPos[];
    int elementId = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementId >= numElements) return;

    float4 elementPosMass = make_float4(body.x[elementId], body.y[elementId], body.z[elementId], body.w[elementId]);
    float3 elementSpeed = make_float3(body.vx[elementId], body.vy[elementId], body.vz[elementId]);
    float3 elementForce = make_float3(0, 0, 0);

    for (int tile = 0; tile < gridDim.x; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (threadIdx.x < blockDim.x && idx < numElements) {
            sharedPos[threadIdx.x] = make_float4(body.x[idx], body.y[idx], body.z[idx], body.w[idx]);
        }
        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i) {
            int index = tile * blockDim.x + i;
            if (index < numElements && index != elementId) {
                // Compute interaction 
                bodyBodyInteraction(elementPosMass, sharedPos[i], elementForce);
            }
        }
    }

    // Update velocity 
    body.vx[elementId] += elementForce.x/elementPosMass.w * TIMESTEP;
    body.vy[elementId] += elementForce.y/elementPosMass.w * TIMESTEP;
    body.vz[elementId] += elementForce.z/elementPosMass.w * TIMESTEP;
}

//
// n-Body Kernel to update the position
// Neended to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, Body_t *bodies)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		bodies->x[elementId] += bodies->vx[elementId] * TIMESTEP;
		bodies->y[elementId] += bodies->vy[elementId] * TIMESTEP;
		bodies->z[elementId] += bodies->vz[elementId] * TIMESTEP;
	}
}

// Function to initialize particle data on the host
void initializeParticles(Body_t &body, int numParticles) {
    body.x = (float *)malloc(numParticles * sizeof(float));
    body.y = (float *)malloc(numParticles * sizeof(float));
    body.z = (float *)malloc(numParticles * sizeof(float));
    body.w = (float *)malloc(numParticles * sizeof(float));
    body.vx = (float *)malloc(numParticles * sizeof(float));
    body.vy = (float *)malloc(numParticles * sizeof(float));
    body.vz = (float *)malloc(numParticles * sizeof(float));

    // Initialize positions, mass, and velocities with random values
    for (int i = 0; i < numParticles; i++) {
        body.x[i] = rand() / (float)RAND_MAX * 100.0f;
        body.y[i] = rand() / (float)RAND_MAX * 100.0f;
        body.z[i] = rand() / (float)RAND_MAX * 100.0f;
        body.w[i] = rand() / (float)RAND_MAX + 1.0f;  // Avoid zero mass
        body.vx[i] = 0.0f;
        body.vy[i] = 0.0f;
        body.vz[i] = 0.0f;
    }
}

// Function to free particle data on the host
void freeParticles(Body_t &body) {
    free(body.x);
    free(body.y);
    free(body.z);
    free(body.w);
    free(body.vx);
    free(body.vy);
    free(body.vz);
}

//
// Main
//
int main(int argc, char *argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);
	}

	// SOA
	// Allocate and initialize particles on the host
    Body_t h_particles;
    initializeParticles(h_particles, numElements);

	//
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
		gridSize = 0,
		numIterations = 0;

	// Number of Iterations
	chCommandLineGet<int>(&numIterations, "i", argc, argv);
	chCommandLineGet<int>(&numIterations, "num-iterations", argc, argv);
	numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize, "t", argc, argv);
	chCommandLineGet<int>(&blockSize, "threads-per-block", argc, argv);
	blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - The number of threads per block is too big" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	gridSize = (CHUNK_SIZE + blockSize - 1) / blockSize;

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;

	bool silent = chCommandLineGetBool("silent", argc, argv);

    Body_t d_body[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    // Allocate memory on the device for each stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_body[i].x, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].y, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].z, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].w, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].vx, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].vy, CHUNK_SIZE * sizeof(float));
        cudaMalloc(&d_body[i].vz, CHUNK_SIZE * sizeof(float));
        cudaStreamCreate(&streams[i]);
    }

	kernelTimer.start();

	// MAIN SIMULATION LOOP
	for (int i = 0; i < numIterations; i++)
	{	
        for (int i = 0; i < numElements; i += CHUNK_SIZE * NUM_STREAMS) {
            for (int s = 0; s < NUM_STREAMS; s++) {
                int offset = i + s * CHUNK_SIZE;
                if (offset >= numElements) break;

                int currentChunkSize = min(CHUNK_SIZE, numElements - offset);

                // Asynchronous memory copy to device
                cudaMemcpyAsync(d_body[s].x, h_particles.x + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].y, h_particles.y + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].z, h_particles.z + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].w, h_particles.w + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].vx, h_particles.vx + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].vy, h_particles.vy + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
                cudaMemcpyAsync(d_body[s].vz, h_particles.vz + offset, currentChunkSize * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

                // Launch kernel
                sharedNbody_Kernel<<<gridSize, blockSize, blockSize * sizeof(float4), streams[s]>>>(currentChunkSize, d_body[s]);
                updatePosition_Kernel<<<gridSize, blockSize, 0, streams[s]>>>(currentChunkSize, &d_body[s]);

                // Asynchronous memory copy back to host
                cudaMemcpyAsync(h_particles.x + offset, d_body[s].x, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.y + offset, d_body[s].y, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.z + offset, d_body[s].z, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.w + offset, d_body[s].w, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.vx + offset, d_body[s].vx, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.vy + offset, d_body[s].vy, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
                cudaMemcpyAsync(h_particles.vz + offset, d_body[s].vz, currentChunkSize * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
            }

            // Synchronize streams
            for (int s = 0; s < NUM_STREAMS; s++) {
                cudaStreamSynchronize(streams[s]);
            }
        }

		// Print some elements
		if (!silent)
		{
			printElement(&h_particles, 0, i + 1);
		}
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimer.stop();

    // Free host memory
    freeParticles(h_particles);

	// Free device memory and destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_body[i].x);
        cudaFree(d_body[i].y);
        cudaFree(d_body[i].z);
        cudaFree(d_body[i].w);
        cudaFree(d_body[i].vx);
        cudaFree(d_body[i].vy);
        cudaFree(d_body[i].vz);
        cudaStreamDestroy(streams[i]);
    }

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Num Iterations: " << numIterations << std::endl
			  << "***    Threads per block: " << blockSize << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
			  << " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
			  << " GB/s" << std::endl
			  << "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
			  << " ms" << std::endl
			  << "***" << std::endl;

	return 0;
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  << std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "    Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "    Number of elements (particles)" << std::endl
			  << "" << std::endl
			  << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
			  << "    Number of iterations" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
			  << std::endl
			  << "    The number of threads per block" << std::endl
			  << "" << std::endl
			  << "  --silent"
			  << std::endl
			  << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
			  << "" << std::endl;
}

//
// Print one element
//
void printElement(Body_t *particles, int elementId, int iteration)
{
	float4 posMass = make_float4(particles->x[elementId], particles->y[elementId], particles->z[elementId], particles->w[elementId]);
	float3 velocity = make_float3(particles->vx[elementId], particles->vy[elementId], particles->vz[elementId]);

	std::cout << "***" << std::endl
			  << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
			  << "*** velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
			  << "*** Mass: <"
			  << std::setw(11) << std::setprecision(9) << posMass.w << "> [kg]" << std::endl
			  << "***" << std::endl;
}


