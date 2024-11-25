#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void checkCuda(cudaError_t err, bool exitOnErr = true) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        std::fflush(stderr);
        if (exitOnErr) {
            std::exit(1);
        }
    }
}

__global__ void measureSharedMemoryStride(volatile float *time, int stride, int iterations, int sharedMemorySize) {
    extern __shared__ volatile float sharedMem[];

    int tid = threadIdx.x;

    // Initialize shared memory
    if (tid < sharedMemorySize) {
        sharedMem[tid] = tid * 1.0f;
    }
    __syncthreads();

    float value = 0.0f;
    unsigned long long start_time, end_time;

    // Start the clock
    start_time = clock64();

    // Access shared memory with a stride
    for (int i = 0; i < iterations; i++) {
        value += sharedMem[(tid * stride) % sharedMemorySize];
    }

    // Stop the clock
    end_time = clock64();

    // Save clock count
    time[tid] = (float)(end_time - start_time);

    __syncthreads();
    
}

int main() {
    const int sharedMemorySize = 1024; // Size of shared memory array
    const int iterations = 10000;     // Number of iterations for stable results
    const int maxStride = 64;         // Maximum stride to test
    const int tpb = 128;              // Threads per block
    const int gridSize = 1;           // Single block for simplicity

    float *d_results, *h_results;
    cudaMalloc(&d_results, maxStride * sizeof(float));
    h_results = (float *)malloc(maxStride * sizeof(float));

    for (int stride = 1; stride <= maxStride; stride++) {
        measureSharedMemoryStride<<<gridSize, tpb, sharedMemorySize * sizeof(float)>>>(d_results, stride, iterations, sharedMemorySize);
        cudaMemcpy(&h_results[stride - 1], d_results, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Output results
    for (int stride = 1; stride <= maxStride; stride++) {
        printf("%d, %f\n", stride, h_results[stride - 1]);
    }

    // Free resources
    cudaFree(d_results);
    free(h_results);

    return 0;
}