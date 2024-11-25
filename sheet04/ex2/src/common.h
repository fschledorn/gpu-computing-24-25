#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "chTimer.hpp"

void checkCuda(cudaError_t err, bool exitOnErr = true) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        std::fflush(stderr);
        if (exitOnErr) {
            std::exit(1);
        }
    }
}

template <typename F>
double timeFn(F f, int iterations = 1024) {
    ChTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    timer.stop();
    return timer.getTime(iterations);
}
