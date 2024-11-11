/*************************************************************************************************
 *
 *        HAWAII, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

#include <assert.h>
#include <chCommandLine.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <chTimer.hpp>
#include <cstdlib>
#include <iomanip>
#include <iostream>

const static int DEFAULT_MEM_SIZE = 10 * 1024 * 1024;  // 10 MB
const static int DEFAULT_NUM_ITERATIONS = 1000;
const static int DEFAULT_BLOCK_DIM = 128;
const static int DEFAULT_GRID_DIM = 16;

//
// Function Prototypes
//
void printHelp(char*);

//
// Kernel Wrappers
//

extern void globalMemCoalescedKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, size_t size);
extern void globalMemStrideKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, int stride);
extern void globalMemOffsetKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, size_t offset);

void checkCuda(cudaError_t cudaError) {
    if (cudaError != cudaSuccess) {
        std::cout << "\033[31m***" << std::endl
                  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                  << std::endl
                  << "***\033[0m" << std::endl;

        std::exit(-1);
    }
}

//
// Main
//
int main(int argc, char* argv[]) {
    // Show Help
    bool optShowHelp = chCommandLineGetBool("h", argc, argv);
    if (!optShowHelp)
        optShowHelp = chCommandLineGetBool("help", argc, argv);

    if (optShowHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    ChTimer memCpyH2DTimer, memCpyD2HTimer, memCpyD2DTimer;
    ChTimer kernelTimer;

    //
    // Get kernel launch parameters and configuration
    //
    int optNumIterations = 0,
        optBlockSize = 0,
        optGridSize = 0;

    // Number of Iterations
    chCommandLineGet<int>(&optNumIterations, "i", argc, argv);
    chCommandLineGet<int>(&optNumIterations, "iterations", argc, argv);
    optNumIterations = (optNumIterations != 0) ? optNumIterations : DEFAULT_NUM_ITERATIONS;

    // Block Dimension / Threads per Block
    chCommandLineGet<int>(&optBlockSize, "t", argc, argv);
    chCommandLineGet<int>(&optBlockSize, "threads-per-block", argc, argv);
    optBlockSize = optBlockSize != 0 ? optBlockSize : DEFAULT_BLOCK_DIM;

    if (optBlockSize > 1024) {
        std::cerr << "\033[31m***"
                  << "*** Error - The number of threads per block is too big"
                  << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    // Grid Dimension
    chCommandLineGet<int>(&optGridSize, "g", argc, argv);
    chCommandLineGet<int>(&optGridSize, "grid-dim", argc, argv);
    optGridSize = optGridSize != 0 ? optGridSize : DEFAULT_GRID_DIM;

    // Sync after each Kernel Launch
    bool optSynchronizeKernelLaunch = chCommandLineGetBool("y", argc, argv);
    if (!optSynchronizeKernelLaunch)
        optSynchronizeKernelLaunch = chCommandLineGetBool("synchronize-kernel", argc, argv);

    int optStride = 1;  // default stride for global-stride test
    chCommandLineGet<int>(&optStride, "stride", argc, argv);

    int optOffset = 0;  // default offset for global-stride test
    chCommandLineGet<int>(&optOffset, "offset", argc, argv);

    // Allocate Memory (take optStride resp. optOffset into account)
    int optMemorySize = 0;

    if (chCommandLineGetBool("global-stride", argc, argv)) {
        // determine memory size from kernel launch parameters and stride
        optMemorySize = optGridSize * optBlockSize * optStride * sizeof(int);
        std::cerr << "*** Ignoring size parameter; using kernel launch parameters and stride, size=" << optMemorySize << std::endl;
    } else if (chCommandLineGetBool("global-offset", argc, argv)) {
        // determine memory size from kernel launch parameters and stride
        optMemorySize = optGridSize * optBlockSize * sizeof(int);
        std::cerr << "*** Ignoring size parameter; using kernel launch parameters and offset, size=" << optMemorySize << std::endl;
    } else {
        // determine memory size from parameters
        chCommandLineGet<int>(&optMemorySize, "s", argc, argv);
        chCommandLineGet<int>(&optMemorySize, "size", argc, argv);
        optMemorySize = optMemorySize != 0 ? optMemorySize : DEFAULT_MEM_SIZE;
    }

    //
    // Host Memory
    //
    int* h_memoryA = NULL;
    int* h_memoryB = NULL;
    bool optUsePinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!optUsePinnedMemory)
        optUsePinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);

    if (!optUsePinnedMemory) {  // Pageable
        h_memoryA = static_cast<int*>(malloc(static_cast<size_t>(optMemorySize)));
        h_memoryB = static_cast<int*>(malloc(static_cast<size_t>(optMemorySize)));
    } else {  // Pinned
        checkCuda(cudaMallocHost(&h_memoryA, optMemorySize));
        checkCuda(cudaMallocHost(&h_memoryB, optMemorySize));
    }

    //
    // Device Memory
    //
    int* d_memoryA = NULL;
    int* d_memoryB = NULL;
    checkCuda(cudaMalloc((void**)&d_memoryA, static_cast<size_t>(optMemorySize)));
    checkCuda(cudaMalloc((void**)&d_memoryB, static_cast<size_t>(optMemorySize)));

    if (!h_memoryA || !h_memoryB || !d_memoryA || !d_memoryB) {
        std::cerr << h_memoryA << " " << h_memoryB << " " << d_memoryA << " " << d_memoryB << "\n";
        std::cerr << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }

    //
    // Copy
    //
    int optMemCpyIterations = 0;
    chCommandLineGet<int>(&optMemCpyIterations, "im", argc, argv);
    chCommandLineGet<int>(&optMemCpyIterations, "memory-copy-iterations", argc, argv);
    optMemCpyIterations = optMemCpyIterations != 0 ? optMemCpyIterations : 1;

    // Host To Device
    memCpyH2DTimer.start();
    for (int i = 0; i < optMemCpyIterations; i++) {
        checkCuda(cudaMemcpy(d_memoryA, h_memoryA, optMemorySize, cudaMemcpyHostToDevice));
        cudaDeviceSynchronize();
    }
    memCpyH2DTimer.stop();

    // Device To Device
    memCpyD2DTimer.start();
    for (int i = 0; i < optMemCpyIterations; i++) {
        checkCuda(cudaMemcpy(d_memoryB, d_memoryA, optMemorySize, cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
    }
    memCpyD2DTimer.stop();

    // Device To Host
    memCpyD2HTimer.start();
    for (int i = 0; i < optMemCpyIterations; i++) {
        checkCuda(cudaMemcpy(h_memoryB, d_memoryB, optMemorySize, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    memCpyD2HTimer.stop();

    //
    // Global Memory Tests
    //
    globalMemStrideKernel_Wrapper(optGridSize, optBlockSize, d_memoryB, d_memoryA, optStride);
    cudaDeviceSynchronize();
    kernelTimer.start();
    for (int i = 0; i < optNumIterations; i++) {
        //
        // Launch Kernel
        //
        if (chCommandLineGetBool("global-coalesced", argc, argv)) {
            globalMemCoalescedKernel_Wrapper(optGridSize, optBlockSize, d_memoryB, d_memoryA, optMemorySize / sizeof(int));
        } else if (chCommandLineGetBool("global-stride", argc, argv)) {
            globalMemStrideKernel_Wrapper(optGridSize, optBlockSize, d_memoryB, d_memoryA, optStride);
        } else if (chCommandLineGetBool("global-offset", argc, argv)) {
            globalMemOffsetKernel_Wrapper(optGridSize, optBlockSize, d_memoryB, d_memoryA, optOffset);
        } else {
            break;
        }

        if (optSynchronizeKernelLaunch) {  // Synchronize after each kernel launch
            cudaDeviceSynchronize();
            checkCuda(cudaGetLastError());
        }

    }

    // Mandatory synchronize after all kernel launches
    cudaDeviceSynchronize();
    kernelTimer.stop();
    checkCuda(cudaGetLastError());

    if (chCommandLineGetBool("global-coalesced", argc, argv)) {
        std::printf("%d,%d,%d,%0.4f\n",
                    optMemorySize, optGridSize, optBlockSize,
                    1e-9 * kernelTimer.getBandwidth(optMemorySize, optNumIterations));
    } else if (chCommandLineGetBool("global-stride", argc, argv)) {
        std::printf("%d,%d,%0.4f\n", optMemorySize, optStride, 1e-9 * kernelTimer.getBandwidth(optMemorySize, optNumIterations));
    } else if (chCommandLineGetBool("global-offset", argc, argv)) {
        std::printf("%d,%d,%0.4f\n", optMemorySize, optOffset, 1e-9 * kernelTimer.getBandwidth(optMemorySize, optNumIterations));
    } else {
        std::printf(
            "%d,%.4f,%.4f,%.4f\n",
            optMemorySize,
            1e-9 * memCpyH2DTimer.getBandwidth(optMemorySize, optMemCpyIterations),
            1e-9 * memCpyD2HTimer.getBandwidth(optMemorySize, optMemCpyIterations),
            1e-9 * memCpyD2DTimer.getBandwidth(optMemorySize, optMemCpyIterations));
    }
}

void printHelp(char* programName) {
    std::cout
        << "Usage: " << std::endl
        << "  " << programName << " [-p] [-s <memory_size>] [-i <num_iterations>]" << std::endl
        << "                [-t <threads_per_block>] [-g <blocks_per_grid]" << std::endl
        << "                [-m <memory-copy-iterations>] [-y] [-stride <stride>] [-offset <offset>]" << std::endl
        << "  --global-{coalesced|stride|offset}" << std::endl
        << "    Run kernel analyzing global memory performance" << std::endl
        << "  -p|--pinned-memory" << std::endl
        << "    Use pinned Memory instead of pageable memory" << std::endl
        << "  -y|--synchronize-kernel" << std::endl
        << "    Synchronize device after each kernel launch" << std::endl
        << "  -s <memory_size>|--size <memory_size>" << std::endl
        << "    The amount of memory to allcate" << std::endl
        << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" << std::endl
        << "    The number of threads per block" << std::endl
        << "  -g <blocks_per_grid>|--grid-dim <blocks_per_grid>" << std::endl
        << "     The number of blocks per grid" << std::endl
        << "  -i <num_iterations>|--iterations <num_iterations>" << std::endl
        << "     The number of iterations to launch the kernel" << std::endl
        << "  --im <memory-copy-iterations>|--memory-iterations <memory-copy-iterations>" << std::endl
        << "     The number of times the memory will be copied. Use this to get more stable results." << std::endl
        << "  --stride <stride>" << std::endl
        << "     Stride parameter for global-stride test. Not that size parameter is ignored then." << std::endl
        << "  --offset <offset>" << std::endl
        << "     Offset parameter for global-offset test. Not that size parameter is ignored then." << std::endl
        << "" << std::endl;
}
