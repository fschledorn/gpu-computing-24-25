#include <cstdio>

#include "common.h"
#include <iostream>
#include <fstream>

#include <cuda.h>

// how many times to overwrite the complete shared memory. this should help amortize launch delays.
// may actually be unnecessary, but i'm hoping for cleaner data this way.
#define REPETITIONS 4
// optimal thread size for ex2 part 2.
#define OPT_THREAD_SIZE 1024
// maximum grid size to try (with the optimal thread count). make sure this doesn't exceed signed int or KABOOM!
#define MAX_GRID_SIZE 512

__global__ void globalToShmemKernel(volatile float arr[], int elems) {
    // use volatile to prevent the compiler from optimizing away the reads
    extern __shared__ volatile float shmem[];

    int globalBase = blockDim.x * blockIdx.x;
    int stride = blockDim.x;

    for (int add = 0; add < REPETITIONS; ++add) {
        for (int offset = 0; offset < elems; offset += stride) {
            int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elems) {
                shmem[idx] = arr[globalBase + idx] + ((float) add);
            }
        }
    }
}

void globalToShmemWrapper(int gridDim, int blockDim, float arr[], std::size_t elems) {
    globalToShmemKernel<<<gridDim, blockDim, elems * sizeof(float)>>>(arr, elems);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}

__global__ void shmemToGlobalKernel(volatile float arr[], int elems) {
    // use volatile to prevent the compiler from optimizing away the reads
    extern __shared__ volatile float shmem[];

    int globalBase = blockDim.x * blockIdx.x;
    int stride = blockDim.x;
    for (int add = 0; add < REPETITIONS; ++add) {
        for (int offset = 0; offset < elems; offset += stride) {
            int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elems) {
                arr[globalBase + idx] = shmem[idx] + ((float) add);
            }
        }
    }
}

void shmemToGlobalWrapper(int gridDim, int blockDim, float arr[], std::size_t elems) {
    shmemToGlobalKernel<<<gridDim, blockDim, elems * sizeof(float)>>>(arr, elems);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}


__global__ void shmemToRegKernel(int elems) {
    // use volatile to prevent the compiler from optimizing away the reads
    extern __shared__ volatile float shmem[];

    int globalBase = blockDim.x * blockIdx.x;
    int stride = blockDim.x;
    volatile float myVar = 1.0f;
    for (int add = 0; add < REPETITIONS; ++add) {
        for (int offset = 0; offset < elems; offset += stride) {
            int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elems) {
                myVar = shmem[idx] + ((float) add);
            }
        }
    }
}

void shmemToRegWrapper(int gridDim, int blockDim, std::size_t elems) {
    shmemToRegKernel<<<gridDim, blockDim, elems * sizeof(float)>>>(elems);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}

__global__ void regToShmemKernel(int elems) {
    // use volatile to prevent the compiler from optimizing away the reads
    extern __shared__ volatile float shmem[];

    int globalBase = blockDim.x * blockIdx.x;
    int stride = blockDim.x;
    volatile float myVar = 1.0f;
    for (int add = 0; add < REPETITIONS; ++add) {
        for (int offset = 0; offset < elems; offset += stride) {
            int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < elems) {
                shmem[idx] = myVar + ((float) add);
            }
        }
    }
}

void regToShmemWrapper(int gridDim, int blockDim, std::size_t elems) {
    regToShmemKernel<<<gridDim, blockDim, elems * sizeof(float)>>>(elems);
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaGetLastError());
}

void timeGlobalToShmem() {
    std::ofstream log("data/global_to_shmem.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/global_to_shmem.csv)");
    }
    log << std::defaultfloat;

    for (int i = 1; i <= 48; i++) {
        std::size_t shmemSize = i * 1024;
        std::size_t elems = shmemSize / sizeof(float);
        float* globalMem;
        checkCuda(cudaMalloc(&globalMem, shmemSize));

        log << shmemSize;
        for (int threads = 1; threads <= 1024; threads *= 4) {
            double t = timeFn([&]() { globalToShmemWrapper(1, threads, globalMem, elems); });
            double throughput = ((double) shmemSize) / t * 1e-9 * ((double) REPETITIONS);
            // std::printf(",%g", t);
            log << ',' << throughput;
            checkCuda(cudaDeviceSynchronize());
        }
        log << std::endl;
    }
}

void timeShmemToGlobal() {
    std::ofstream log("data/shmem_to_global.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/shmem_to_global.csv)");
    }
    log << std::defaultfloat;

    for (int i = 1; i <= 48; i++) {
        std::size_t shmemSize = i * 1024;
        std::size_t elems = shmemSize / sizeof(float);
        float* globalMem;
        checkCuda(cudaMalloc(&globalMem, shmemSize));

        log << shmemSize;
        for (int threads = 1; threads <= 1024; threads *= 4) {
            double t = timeFn([&]() { shmemToGlobalWrapper(1, threads, globalMem, elems); });
            double throughput = ((double) shmemSize) / t * 1e-9 * ((double) REPETITIONS);
            // std::printf(",%g", t);
            log << ',' << throughput;
            checkCuda(cudaDeviceSynchronize());
        }
        log << std::endl;
    }
}

void timeShmemToGlobalWithGrid() {
    std::ofstream log("data/shmem_to_global_variable_gridsize.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/shmem_to_global_variable_gridsize.csv)");
    }
    log << std::defaultfloat;

    // contant shmemSize for this test; use max and hope for cleaner results that way.
    std::size_t shmemSize = 48 * 1024;
    std::size_t elems = shmemSize / sizeof(float);

    // maximum grid size to use; we expect some dropoff after SM count of an rtx 2080, which is 46.
    int maxGridSize = MAX_GRID_SIZE;

    // get global memory large enough to suit all tests
    float* globalMem;
    checkCuda(cudaMalloc(&globalMem, shmemSize * maxGridSize));

    for (int gridSize = 1; gridSize <= maxGridSize; ++gridSize) {
        log << gridSize;
        int threads = OPT_THREAD_SIZE;
        double t = timeFn([&]() { globalToShmemWrapper(gridSize, threads, globalMem, elems); });
        double throughput = ((double) shmemSize * gridSize) / t * 1e-9 * ((double) REPETITIONS);
        // std::printf(",%g", t);
        log << ',' << throughput;
        checkCuda(cudaDeviceSynchronize());


        log << std::endl;
    }
}

void timeGlobalToShmemWithGrid() {
    std::ofstream log("data/global_to_shmem_variable_gridsize.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/global_to_shmem_variable_gridsize.csv)");
    }
    log << std::defaultfloat;

    // contant shmemSize for this test; use max and hope for cleaner results that way.
    std::size_t shmemSize = 48 * 1024;
    std::size_t elems = shmemSize / sizeof(float);

    // maximum grid size to use; we expect some dropoff after SM count of an rtx 2080, which is 46
    int maxGridSize = MAX_GRID_SIZE;

    // get global memory large enough to suit all tests
    float* globalMem;
    checkCuda(cudaMalloc(&globalMem, shmemSize * maxGridSize));

    for (int gridSize = 1; gridSize <= maxGridSize; ++gridSize) {
        log << gridSize;
        int threads = OPT_THREAD_SIZE;
        double t = timeFn([&]() { shmemToGlobalWrapper(gridSize, threads, globalMem, elems); });
        double throughput = ((double) shmemSize * gridSize) / t * 1e-9 * ((double) REPETITIONS);
        // std::printf(",%g", t);
        log << ',' << throughput;
        checkCuda(cudaDeviceSynchronize());

        log << std::endl;
    }
}

void timeShmemToReg() {
    std::ofstream log("data/shmem_to_reg.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/shmem_to_reg.csv)");
    }
    log << std::defaultfloat;

    for (int i = 1; i <= 48; i++) {
        std::size_t shmemSize = i * 1024;
        std::size_t elems = shmemSize / sizeof(float);

        log << shmemSize;
        for (int threads = 1; threads <= 1024; threads *= 4) {
            double t = timeFn([&]() { shmemToRegWrapper(1, threads, elems); });
            double throughput = ((double) shmemSize) / t * 1e-9 * ((double) REPETITIONS);
            // std::printf(",%g", t);
            log << ',' << throughput;
            checkCuda(cudaDeviceSynchronize());
        }
        log << std::endl;
    }
}

void timeRegToShmem() {
    std::ofstream log("data/reg_to_shmem.csv");
    if (log.fail()) {
        throw new std::runtime_error("failed to open output log (data/reg_to_shmem.csv)");
    }
    log << std::defaultfloat;

    for (int i = 1; i <= 48; i++) {
        std::size_t shmemSize = i * 1024;
        std::size_t elems = shmemSize / sizeof(float);

        log << shmemSize;
        for (int threads = 1; threads <= 1024; threads *= 4) {
            double t = timeFn([&]() { regToShmemWrapper(1, threads, elems); });
            double throughput = ((double) shmemSize) / t * 1e-9 * ((double) REPETITIONS);
            // std::printf(",%g", t);
            log << ',' << throughput;
            checkCuda(cudaDeviceSynchronize());
        }
        log << std::endl;
    }
}


int main() {
    /*
    int device = -1;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp props;
    checkCuda(cudaGetDeviceProperties(&props, device));
    std::cout << "shared mem: " << props.sharedMemPerMultiprocessor << std::endl;
    */

    std::cout << "Timing global to shared" << std::endl;
    timeGlobalToShmem();
    std::cout << "Timing shared to global" << std::endl;
    timeShmemToGlobal();
    std::cout << "Timing each with large grid sizes: global to shared" << std::endl;
    timeGlobalToShmemWithGrid();
    std::cout << "Timing each with large grid sizes: shared to global" << std::endl;
    timeShmemToGlobalWithGrid();

    std::cout << "Timing read from shared memory (to registers)" << std::endl;
    timeShmemToReg();

    std::cout << "Timing write to shared memory (to registers)" << std::endl;
    timeRegToShmem();
}
