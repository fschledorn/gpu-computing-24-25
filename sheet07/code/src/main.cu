#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>
#include <hdf5.h>

const static int DEFAULT_NUM_ELEMENTS = 1024;
const static int DEFAULT_NUM_ITERATIONS = 100;
const static int DEFAULT_BLOCK_DIM = 32;

const static float TIMESTEP = 1e-6;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)

//
// Structures
//
// Here with two AOS (arrays of structures).
//
struct Body_t
{
	float4 posMass;	 /* x = x */
					 /* y = y */
					 /* z = z */
					 /* w = Mass */
	float3 velocity; /* x = v_x*/
					 /* y = v_y */
					 /* z= v_z */

	Body_t() : posMass(make_float4(0, 0, 0, 0)), velocity(make_float3(0, 0, 0)) {}
};

//
// Function Prototypes
//
void printHelp(char *);
void printElement(Body_t *, int, int);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

//
// Calculate the forces between two bodies
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
	float distance = getDistance(bodyA, bodyB);

	if (distance == 0)
		return;

	// Calc Force
	float3 direction = make_float3(bodyB.x - bodyA.x, bodyB.y - bodyA.y, bodyB.z - bodyA.z);

	direction.x /= distance;
	direction.y /= distance;
	direction.z /= distance;

	force.x = -GAMMA * bodyA.w * bodyB.w / (distance * distance + 1e-6)*direction.x;
	force.y = -GAMMA * bodyA.w * bodyB.w / (distance * distance + 1e-6)*direction.y;
	force.z = -GAMMA * bodyA.w * bodyB.w / (distance * distance + 1e-6)*direction.z;
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3 &currentSpeed, float3 force)
{
	// Calculate the new velocity of a particle
	float3 acceleration = make_float3(force.x / mass, force.y / mass, force.z / mass);
	currentSpeed.x += acceleration.x * TIMESTEP;
	currentSpeed.y += acceleration.y * TIMESTEP;
	currentSpeed.z += acceleration.z * TIMESTEP;
}

//
// n-Body Kernel for the speed calculation
//
__global__ void
simpleNbody_Kernel(int numElements, Body_t *body)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;

	if (elementId < numElements)
	{
		elementPosMass = body[elementId].posMass;
		elementSpeed = body[elementId].velocity;
		elementForce = make_float3(0, 0, 0);

		for (int i = 0; i < numElements; i++)
		{
			if (i != elementId)
			{
				bodyBodyInteraction(elementPosMass, body[i].posMass, elementForce);
			}
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		body[elementId].velocity = elementSpeed;
	}
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
		// Update position of a particle
		float3 velocity = bodies[elementId].velocity;
		bodies[elementId].posMass.x += velocity.x * TIMESTEP;
		bodies[elementId].posMass.y += velocity.y * TIMESTEP;
		bodies[elementId].posMass.z += velocity.z * TIMESTEP;
	}
}

void saveToHDF5(const char *filename, Body_t *bodies, int numElements)
{
    // Define the dataset dimensions explicitly
    hsize_t dims[2] = {static_cast<hsize_t>(numElements), 4};
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t space_id = H5Screate_simple(2, dims, NULL);

    hid_t dataset_id = H5Dcreate(file_id, "positions", H5T_NATIVE_FLOAT, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Prepare data for writing
    float *data = new float[numElements * 4];
    for (int i = 0; i < numElements; i++)
    {
        data[i * 4 + 0] = bodies[i].posMass.x;
        data[i * 4 + 1] = bodies[i].posMass.y;
        data[i * 4 + 2] = bodies[i].posMass.z;
        data[i * 4 + 3] = bodies[i].posMass.w;
    }

    // Write data to the dataset
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // Clean up
    delete[] data;
    H5Dclose(dataset_id);
    H5Sclose(space_id);
    H5Fclose(file_id);
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

	Body_t *h_particles;
	if (!pinnedMemory)
	{
		// Pageable
		h_particles = static_cast<Body_t *>(malloc(static_cast<size_t>(numElements * sizeof(*h_particles))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_particles, static_cast<size_t>(numElements * sizeof(*h_particles)));
	}

	// Init Particles
	//	srand(static_cast<unsigned>(time(0)));
	srand(0); // Always the same random numbers
	for (int i = 0; i < numElements; i++)
	{
		h_particles[i].posMass.x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
		h_particles[i].posMass.y = 1e-8 * static_cast<float>(rand()); // increase the position changes
		h_particles[i].posMass.z = 1e-8 * static_cast<float>(rand()); // and the velocity
		h_particles[i].posMass.w = 1e4 * static_cast<float>(rand());
		h_particles[i].velocity.x = 0.0f;
		h_particles[i].velocity.y = 0.0f;
		h_particles[i].velocity.z = 0.0f;
	}

	printElement(h_particles, 0, 0);

	// Device Memory
	Body_t *d_particles;
	cudaMalloc(&d_particles, static_cast<size_t>(numElements * sizeof(*d_particles)));

	if (h_particles == NULL || d_particles == NULL)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - Memory allocation failed" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_particles, h_particles, static_cast<size_t>(numElements * sizeof(*d_particles)), cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

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

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;

	bool silent = chCommandLineGetBool("silent", argc, argv);

	kernelTimer.start();

	// MAIN SIMULATION LOOP
	for (int i = 0; i < numIterations; i++)
	{	
		// Launch Kernel
		simpleNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles);
		updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles);

		// Copy Data back to Host
		// cudaMemcpy(h_particles, d_particles, static_cast<size_t>(numElements * sizeof(*h_particles)), cudaMemcpyDeviceToHost);
		// if (!silent)
		// {
		// 	printElement(h_particles, 0, i + 1);
		// }

		// // Save to HDF5
		// char filename[256];
        // sprintf(filename, "iteration_%d.h5", i);
        // saveToHDF5(filename, h_particles, numElements);
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

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_particles, d_particles, static_cast<size_t>(numElements * sizeof(*d_particles)), cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	// Free Memory
	if (!pinnedMemory)
	{
		free(h_particles);
	}
	else
	{
		cudaFreeHost(h_particles);
	}

	cudaFree(d_particles);

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
	float4 posMass = particles[elementId].posMass;
	float3 velocity = particles[elementId].velocity;

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


