#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "common.hpp"

namespace cg = cooperative_groups;

float sumreduce_cpu(float* arr, size_t N) {
    return std::accumulate(arr, arr + N, 0.0f);
}

struct Result {
    float value;
    double time;
};

namespace ex3 {
// expects `gridDim.y=gridDim.z=blockDim.y=1` and
// `sizeof(shmem)=sizeof(float)*blockDim.x`.
//
// `gridDim.x * blockDim.x` must be at least `N`.
//
// `in` must have length at least `N`.
//
// `out` must have length at least `ceil(N / blockDim.x)`, and that many values
// will be written to it, each value being the sum of one block of size
// `blockDim.x` in `in`.
__global__ void sumreduce_kernel(float* out, const float* in, size_t N) {
    extern __shared__ float shmem[];

    size_t tid = threadIdx.x;
    size_t global_offset = blockDim.x * blockIdx.x;

    // either all or none of the threads return here so there's no
    // `__syncthreads()` issues.
    if (global_offset >= N) return;

    size_t segment_size = min(N - global_offset, size_t(blockDim.x));

    if (tid < segment_size) {
        shmem[tid] = in[global_offset + tid];
    }

    __syncthreads();

    // `segment_size` is the same value for each thread in a block, so all
    // threads will do the same number of iterations and the `__syncthreads()`
    // will behave correctly.
    for (size_t stride = 1; stride < segment_size; stride *= 2) {
        if ((tid & (2 * stride - 1)) == 0 && (tid + stride) < segment_size) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = shmem[0];
    }
}

Result sumreduce(const float* arr, size_t N, size_t threads_per_block) {
    if (N == 0) {
        return Result{0, 0};
    }

    float *d_A, *d_B;
    checkCuda(cudaMalloc(&d_A, N * sizeof(float)));
    // the second buffer only needs ceil(N/threads_per_block) elements.
    checkCuda(cudaMalloc(&d_B, ceil_div(N, threads_per_block) * sizeof(float)));
    checkCuda(cudaMemcpy(d_A, arr, N * sizeof(float), cudaMemcpyHostToDevice));

    ChTimer t;
    t.start();
    size_t working_set_size = N;
    // NOTE: We can't force
    while (working_set_size > 1) {
        // we need `ceil(size/threads)` blocks to cover the whole working set.
        size_t block_count = ceil_div(working_set_size, threads_per_block);
        size_t shmem_size = threads_per_block * sizeof(float);
        sumreduce_kernel<<<block_count, threads_per_block, shmem_size>>>(
            d_B, d_A, working_set_size);
        checkCuda(cudaGetLastError());

        working_set_size = block_count;
        std::swap(d_A, d_B);
    }
    checkCuda(cudaDeviceSynchronize());
    t.stop();

    float sum = 0;
    checkCuda(cudaMemcpy(&sum, d_A, sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));

    return Result{sum, t.getTime()};
}
}  // namespace ex3

namespace ex4 {
// this uses most of the optimizations presented in lecture 5. in particular, it
// - uses sequential addressing instead of interleaved
// - does the first add during the initial memory load, almost doubling
// throughput
// - unrolls the last warp, requiring no more block-level synchronization.
__global__ void sumreduce_kernel(float* out, const float* in, size_t N) {
    extern __shared__ float shmem[];

    size_t tid = threadIdx.x;
    size_t global_offset = blockDim.x * blockIdx.x * 2;
    size_t i = global_offset + tid;

    size_t initial_load_segment_size =
        min(N - global_offset, size_t(2 * blockDim.x));
    size_t segment_size = min(N - global_offset, size_t(blockDim.x));

    shmem[tid] = 0;
    if (tid < segment_size) {
        shmem[tid] += in[i];
    }
    if (tid + blockDim.x < initial_load_segment_size) {
        shmem[tid] += in[i + blockDim.x];
    }

    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride && (tid + stride) < segment_size) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    // we need these syncwarps, because with volta's independent thread
    // scheduling the shared memory accesses aren't guaranteed to be
    // synchronized even within a single warp, and for some reason this led
    // to wrong results even with the compiler flags supposedly forcing
    // Pascal thread scheduling.
    if (tid < 32 && (tid + 32) < segment_size) shmem[tid] += shmem[tid + 32];
    __syncwarp();
    if (tid < 16 && (tid + 16) < segment_size) shmem[tid] += shmem[tid + 16];
    __syncwarp();
    if (tid < 8 && (tid + 8) < segment_size) shmem[tid] += shmem[tid + 8];
    __syncwarp();
    if (tid < 4 && (tid + 4) < segment_size) shmem[tid] += shmem[tid + 4];
    __syncwarp();
    if (tid < 2 && (tid + 2) < segment_size) shmem[tid] += shmem[tid + 2];
    __syncwarp();
    if (tid < 1 && (tid + 1) < segment_size) shmem[tid] += shmem[tid + 1];
    __syncwarp();

    if (tid == 0) {
        out[blockIdx.x] = shmem[0];
    }
}

Result sumreduce(const float* arr, size_t N, size_t threads_per_block) {
    if (N == 0) {
        return Result{0, 0};
    }

    float *d_A, *d_B;
    checkCuda(cudaMalloc(&d_A, N * sizeof(float)));
    // the second buffer only needs ceil(N/threads_per_block) elements.
    checkCuda(cudaMalloc(&d_B, ceil_div(N, threads_per_block) * sizeof(float)));
    checkCuda(cudaMemcpy(d_A, arr, N * sizeof(float), cudaMemcpyHostToDevice));

    ChTimer t;
    t.start();
    size_t working_set_size = N;
    while (working_set_size > 1) {
        // we need `ceil(size/(2*threads))` blocks to cover the whole
        // working set.
        size_t block_count = ceil_div(working_set_size, 2 * threads_per_block);
        size_t shmem_size = threads_per_block * sizeof(float);
        sumreduce_kernel<<<block_count, threads_per_block, shmem_size>>>(
            d_B, d_A, working_set_size);
        checkCuda(cudaGetLastError());

        working_set_size = block_count;
        std::swap(d_A, d_B);
    }
    checkCuda(cudaDeviceSynchronize());
    t.stop();

    float sum = 0;
    checkCuda(cudaMemcpy(&sum, d_A, sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));

    return Result{sum, t.getTime()};
}
}  // namespace ex4

namespace ex5 {
// this is an even more optimized sum reduction kernel. for large datasets,
// this version saturates device ram bandwidth, so there's no way to further
// speed it up. while some of the optimizations used are listed on the volta
// specific slides in the lecture, all features seem to be available starting
// on kepler series gpus.
//
// optimizations used compared to the kernel for exercise 4:
// - the kernel is templatized over the number of elements that each thread
//   sums up sequentially before doing the reduction. from benchmarking it
//   seems that the optimal number of elements per thread is 8.
// - intra thread block reduction uses `cg::reduce()` to add all values in each warp
//   without touching memory (could also be done using `__shfl_down`). because each
//   block has at most 32 warps, the intermediate sums can then be summed up by a single
//   warp afterwards. thus, each block sum reduction only requires <= 64 shared memory
//   accesses and one thread block synchronization.
// - instead of each block writing one output value, the values are instead summed up
//   atomically into the final result. this way, only one kernel launch is needed for
//   the reduction.


// adds the values of `val` across all threads on a block and returns the
// sum on thread 0. all other threads return unspecified values. this assumes
// that the block size is a multiple of the warp size, i.e. 32.
__device__ float sumreduce_block(float val) {
    // there's at most 1024 threads per block, so by reducing each warp first
    // we need to store at most 32 intermediate results.
    static __shared__ float shmem[32];

    auto block = cg::this_thread_block();
    // partition the thread block into tiles of 32 threads, i.e. warps.
    auto warp = cg::tiled_partition<32>(block);
    size_t warp_id = warp.meta_group_rank();
    size_t num_warps = warp.meta_group_size();
    size_t lane = warp.thread_rank();
    size_t tid = block.thread_rank();

    // sum up the values within each warp and write each intermediate sum into
    // shared memory
    val = cg::reduce(warp, val, cg::plus<float>());
    if (lane == 0) shmem[warp_id] = val;

    // sync the entire thread block so warp 0 sees all shared mem writes
    block.sync();
    // compute the sum of the intermediate sums in warp 0
    val = tid < num_warps ? shmem[tid] : 0.0f;
    if (warp_id == 0) val = cg::reduce(warp, val, cg::plus<float>());
    // thread 0 of the block will return the sum of all values in the block
    return val;
}

// the template parameter `elems_per_thread` controls how many values
// each thread reads in sequentially before starting the sum reduction.
template <size_t elems_per_thread>
__global__ void sumreduce_kernel(float* out, float* in, size_t N) {
    size_t idx =
        size_t(blockDim.x) * blockIdx.x * elems_per_thread + threadIdx.x;

    float val = 0;

#pragma unroll
    for (size_t i = 0; i < elems_per_thread; i++) {
        if ((idx + i * blockDim.x) < N) val += in[idx + i * blockDim.x];
    }

    // sum up all the values in this block.
    val = sumreduce_block(val);
    if (threadIdx.x == 0) atomicAdd(out, val);
}

Result sumreduce(const float* arr, size_t N, size_t threads_per_block) {
    if (N == 0) {
        return Result{0, 0};
    }

    float* dev_mem;
    checkCuda(cudaMalloc(&dev_mem, (N + 1) * sizeof(float)));
    float* d_sum = dev_mem;
    float* d_arr = &dev_mem[1];

    checkCuda(cudaMemset(d_sum, 0, sizeof(float)));
    checkCuda(
        cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice));

    const int elems_per_thread = 8;

    size_t num_blocks = ceil_div(N, elems_per_thread * threads_per_block);
    ChTimer t;
    t.start();
    sumreduce_kernel<elems_per_thread>
        <<<num_blocks, threads_per_block>>>(d_sum, d_arr, N);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    t.stop();

    float sum = 0;
    checkCuda(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(dev_mem));
    return Result{sum, t.getTime()};
}
}  // namespace ex5

template <typename F>
void bench_sumreduce_gpu(F f, const char* path) {
    auto file = open_file(path);
    auto file_ptr = file.get();

    std::vector<float> buffer((1 << 30) / sizeof(float));
    for (size_t i = 0; i < buffer.size(); i++) buffer[i] = 1.0 / float(i + 1);

    for (size_t bytes = 1 << 10; bytes <= (1 << 30); bytes *= 2) {
        for (int threads : {32, 64, 128, 256, 512, 1024}) {
            volatile float sum;
            int iterations = std::max(size_t(32), size_t(1 << 20) / bytes);
            double time = 0;
            for (int i = 0; i < iterations; i++) {
                Result res = f(buffer.data(), bytes / sizeof(float), threads);
                sum = res.value;
                time += res.time;
            }
            time /= iterations;
            double bandwidth = bytes / time * 1e-9;
            std::fprintf(file_ptr, "%zu,%d,%g,%g\n", bytes, threads, time,
                         bandwidth);
            std::fflush(file_ptr);

            float expected =
                sumreduce_cpu(buffer.data(), bytes / sizeof(float));
            if (std::fabs(expected - sum) > 1e-7 * bytes) {
                std::fprintf(
                    stderr,
                    "Mismatch for size %zu and %d threads: CPU=%g, GPU=%g\n",
                    bytes, threads, expected, sum);
                std::fflush(stderr);
            }
        }
    }
}

void run_ex2() {
    auto file = open_file("data/ex2.csv");
    auto file_ptr = file.get();

    std::vector<float> buffer((1 << 30) / sizeof(float));
    for (size_t i = 0; i < buffer.size(); i++) buffer[i] = 1.0 / float(i + 1);

    for (size_t bytes = 1 << 10; bytes <= (1 << 30); bytes *= 2) {
        volatile float sum;
        int iterations = std::max(size_t(32), size_t(1 << 25) / bytes);
        double time = time_fn(
            [&] { sum = sumreduce_cpu(buffer.data(), bytes / sizeof(float)); },
            iterations);
        double bandwidth = bytes / time / double(1 << 30);
        std::fprintf(file_ptr, "%zu,%g,%g\n", bytes, time, bandwidth);
        std::fflush(file_ptr);
    }
}

void run_ex3() { bench_sumreduce_gpu(ex3::sumreduce, "data/ex3.csv"); }

void run_ex4() { bench_sumreduce_gpu(ex4::sumreduce, "data/ex4.csv"); }

void run_ex5() { bench_sumreduce_gpu(ex5::sumreduce, "data/ex5.csv"); }

int main() {
    run_ex2();
    run_ex3();
    run_ex4();
    run_ex5();
}