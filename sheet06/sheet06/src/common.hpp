
#include <cstdio>
#include <cstdlib>
#include "chTimer.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <cstdio>

inline void checkCuda(cudaError_t err, bool exitOnErr = true) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        std::fflush(stderr);
        if (exitOnErr) {
            std::exit(1);
        }
    }
}

inline std::unique_ptr<FILE, decltype(&fclose)> open_file(const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        perror("Opening file");
        std::exit(1);
    }
    return std::unique_ptr<FILE, decltype(&fclose)>{ fp, &fclose };
}

template <typename T>
T ceil_div(T lhs, T rhs) {
    T div = lhs / rhs;
    T rem = lhs % rhs;
    return div + (rem != 0);
}

template <typename F>
double time_fn(F f, int iterations = 1024) {
    ChTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    timer.stop();
    return timer.getTime(iterations);
}