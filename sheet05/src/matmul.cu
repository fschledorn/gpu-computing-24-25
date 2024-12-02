#include <cuda_runtime.h>

#include <cstdio>

#include "common.hpp"

float max_diff(const float* a, const float* b, int n) {
    float d = 0;
    for (int i = 0; i < n; i++) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

void printMat(const float* mat, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::printf("%g\t\t", mat[i * N + j]);
        }
        std::printf("\n");
    }
}

struct Timings {
    double hostToDev = 0;
    double kernelTime = 0;
    double devToHost = 0;
};

void matmul_cpu(int N, float* A, float* B, float* C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void matmul_naive_kernel(int N, const float* A, const float* B,
                                    float* C) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx / N;
    int j = idx % N;
    if (i >= N) return;
    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

Timings matmul_naive(int N, const float* A, const float* B, float* C,
                     int threads_per_block) {
    int elem_count = N * N;
    int mem_size = sizeof(float) * elem_count * 3;
    float* device_mem;
    checkCuda(cudaMalloc(&device_mem, mem_size));

    float* d_A = &device_mem[0 * elem_count];
    float* d_B = &device_mem[1 * elem_count];
    float* d_C = &device_mem[2 * elem_count];

    ChTimer h2d_timer;
    h2d_timer.start();
    checkCuda(
        cudaMemcpy(d_A, A, elem_count * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(d_B, B, elem_count * sizeof(float), cudaMemcpyHostToDevice));
    h2d_timer.stop();

    // block_count = ceil(elem_count / threads_per_block)
    int block_count = (elem_count + threads_per_block - 1) / threads_per_block;
    ChTimer kernel_timer;
    kernel_timer.start();
    matmul_naive_kernel<<<block_count, threads_per_block>>>(N, d_A, d_B, d_C);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    kernel_timer.stop();

    ChTimer d2h_timer;
    d2h_timer.start();
    checkCuda(
        cudaMemcpy(C, d_C, elem_count * sizeof(float), cudaMemcpyDeviceToHost));
    d2h_timer.stop();

    checkCuda(cudaFree(device_mem));

    return Timings{
        h2d_timer.getTime(),
        kernel_timer.getTime(),
        d2h_timer.getTime(),
    };
}

// this must be called with tile_width := blockDim.x == blockDim.y and with 2 *
// sizeof(float) * tile_width * tile_width shared memory
__global__ void matmul_tiled_kernel(int N, const float* A, const float* B,
                                    float* C) {
    int tile_width = blockDim.x;
    int tile_mem = tile_width * tile_width;
    extern __shared__ float shmem[];
    float* As = &shmem[0];
    float* Bs = &shmem[tile_mem];

    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * tile_width + tx;
    int row = blockIdx.y * tile_width + ty;

    float sum = 0;

    // overshoot by one iteration to handle matrix sizes that are not multiples
    // of the tile size.
    for (int m = 0; m < N + tile_width - 1; m += tile_width) {
        // it's important here that the __syncthreads() is outside of the
        // if-blocks, so all threads execute it, as not doing so would be
        // undefined behavior.
        if (m + tx < N) {
            As[ty * tile_width + tx] = A[row * N + m + tx];
        }
        if (m + ty < N) {
            Bs[ty * tile_width + tx] = B[col + (ty + m) * N];
        }
        __syncthreads();

        int max_k = min(tile_width, N - m);
        for (int k = 0; k < max_k; k++) {
            sum += As[ty * tile_width + k] * Bs[k * tile_width + tx];
        }
        __syncthreads();
    }

    if (max(row, col) < N) C[row * N + col] = sum;
}

Timings matmul_tiled(int N, const float* A, const float* B, float* C,
                     int tile_width) {
    int elem_count = N * N;
    int mem_size = sizeof(float) * elem_count * 3;
    int shmem_size = 2 * sizeof(float) * tile_width * tile_width;
    float* device_mem;
    checkCuda(cudaMalloc(&device_mem, mem_size));

    float* d_A = &device_mem[0 * elem_count];
    float* d_B = &device_mem[1 * elem_count];
    float* d_C = &device_mem[2 * elem_count];

    ChTimer h2d_timer;
    h2d_timer.start();
    checkCuda(
        cudaMemcpy(d_A, A, elem_count * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(
        cudaMemcpy(d_B, B, elem_count * sizeof(float), cudaMemcpyHostToDevice));
    h2d_timer.stop();

    // block_count = ceil(N / tile_width)
    int block_count = (N + tile_width - 1) / tile_width;
    ChTimer kernel_timer;
    kernel_timer.start();
    matmul_tiled_kernel<<<dim3(block_count, block_count),
                          dim3(tile_width, tile_width), shmem_size>>>(N, d_A,
                                                                      d_B, d_C);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    kernel_timer.stop();

    ChTimer d2h_timer;
    d2h_timer.start();
    checkCuda(
        cudaMemcpy(C, d_C, elem_count * sizeof(float), cudaMemcpyDeviceToHost));
    d2h_timer.stop();

    checkCuda(cudaFree(device_mem));

    return Timings{
        h2d_timer.getTime(),
        kernel_timer.getTime(),
        d2h_timer.getTime(),
    };
}

void ex2(const char* log_path) {
    auto file = open_file(log_path);
    auto file_ptr = file.get();

    for (int N = 128; N <= 2560; N += 128) {
        int elem_count = N * N;
        int mem_size = sizeof(float) * elem_count * 4;
        float* host_mem;
        checkCuda(cudaMallocHost(&host_mem, mem_size));
        float* A = &host_mem[0 * elem_count];
        float* B = &host_mem[1 * elem_count];
        float* C1 = &host_mem[2 * elem_count];
        float* C2 = &host_mem[3 * elem_count];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = i + j;
                B[i * N + j] = float(i) / (j + 1);
            }
        }

        // launch once to prevent cold launch in the timings
        matmul_naive(N, A, B, C1, 1024);
        if (N <= 1024) {
            matmul_cpu(N, A, B, C2);
            float diff = max_diff(C1, C2, N * N);
            if (diff > 1e-6 * C2[N * (N - 1)]) {
                std::fprintf(stderr, "matmul mismatch for size %d\n", N);
                std::printf("%g\n\n", diff);
                printMat(C1, N);
                std::printf("\n");
                printMat(C2, N);
                std::exit(1);
            }
        }

        for (int t = 1; t <= 1024; t *= 4) {
            int iterations = 64;
            double h2d = 0, kernel = 0, d2h = 0;

            for (int i = 0; i < iterations; i++) {
                Timings timings = matmul_naive(N, A, B, C1, t);
                h2d += timings.hostToDev;
                kernel = timings.kernelTime;
                d2h += timings.devToHost;
            }

            std::fprintf(file_ptr, "%d,%d,%g,%g,%g\n", N, t, h2d / iterations,
                         kernel / iterations, d2h / iterations);
            std::fflush(file_ptr);
        }

        checkCuda(cudaFreeHost(host_mem));
    }
}

void ex3(const char* threads_log, const char* size_log) {
    auto varying_threads = open_file(threads_log);
    auto varying_threads_ptr = varying_threads.get();

    auto varying_size = open_file(size_log);
    auto varying_size_ptr = varying_size.get();

    {
        const int N = 8192;

        printf("using %dx%d matrix size for timing\n", N, N);
        fflush(stdout);

        int elem_count = N * N;
        int mem_size = sizeof(float) * elem_count * 3;
        float* host_mem;
        checkCuda(cudaMallocHost(&host_mem, mem_size));
        float* A = &host_mem[0 * elem_count];
        float* B = &host_mem[1 * elem_count];
        float* C = &host_mem[2 * elem_count];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = i + j;
                B[i * N + j] = float(i) / (j + 1);
            }
        }

        for (int tile_width = 4; tile_width <= 32; tile_width += 4) {
            printf("timing tile width %d\n", tile_width);
            fflush(stdout);

            int iterations = 16;
            double h2d = 0, kernel = 0, d2h = 0;

            for (int i = 0; i < iterations; i++) {
                Timings timings = matmul_tiled(N, A, B, C, tile_width);
                h2d += timings.hostToDev;
                kernel = timings.kernelTime;
                d2h += timings.devToHost;
            }

            std::fprintf(varying_threads_ptr, "%d,%g,%g,%g\n", tile_width * tile_width,
                         h2d / iterations, kernel / iterations,
                         d2h / iterations);
            std::fflush(varying_threads_ptr);
        }

        checkCuda(cudaFreeHost(host_mem));
    }

    printf("checking for correctness!\n");
    fflush(stdout);

    for (int N = 128; N <= 2560; N += 128) {
        int elem_count = N * N;
        int mem_size = sizeof(float) * elem_count * 4;
        float* host_mem;
        checkCuda(cudaMallocHost(&host_mem, mem_size));
        float* A = &host_mem[0 * elem_count];
        float* B = &host_mem[1 * elem_count];
        float* C1 = &host_mem[2 * elem_count];
        float* C2 = &host_mem[3 * elem_count];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = i + j;
                B[i * N + j] = float(i) / (j + 1);
            }
        }

        // launch once to prevent cold launch in the timings
        matmul_tiled(N, A, B, C1, 32);
        if (N <= 512) {
            matmul_cpu(N, A, B, C2);

            float diff = max_diff(C1, C2, N * N);
            if (diff > 1e-6 * C2[N * (N - 1)]) {
                std::fprintf(stderr, "matmul mismatch for size %d\n", N);
                std::printf("%g\n\n", diff);
                printMat(C1, N);
                std::printf("\n");
                printMat(C2, N);
                std::exit(1);
            }
        }

        int iterations = 64;
        double h2d = 0, kernel = 0, d2h = 0;

        for (int i = 0; i < iterations; i++) {
            Timings timings = matmul_tiled(N, A, B, C1, 32);
            h2d += timings.hostToDev;
            kernel = timings.kernelTime;
            d2h += timings.devToHost;
        }

        std::fprintf(varying_size_ptr, "%d,%g,%g,%g\n", N, h2d / iterations,
                     kernel / iterations, d2h / iterations);
        std::fflush(varying_size_ptr);

        checkCuda(cudaFreeHost(host_mem));
    }

    printf("seems correct up to size 512x512 :3\n");
}

int main() {
    // ex2("data/ex2.csv");
    // ex3("data/ex3/varying_threads.csv", "data/ex3/varying_size.csv");
    ex2("data/ex4/naive.csv");
    ex3("data/ex4/varying_threads.csv", "data/ex4/varying_size.csv");
}