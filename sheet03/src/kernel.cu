/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(int* dst, int* src, size_t size)
{
    for (size_t offset = 0; offset < size; offset += size_t(gridDim.x) * size_t(blockDim.x)) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
        if (idx < size) {
            dst[idx] = src[idx];
        }
    }
}

void 
globalMemCoalescedKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, size_t size) {
	globalMemCoalescedKernel<<<gridDim, blockDim, 0>>>(dst, src, size);
}

__global__ void 
globalMemStrideKernel(int* dst, int* src, int stride, int b)
{
    // #*+~#*+~
    // ##**++~~

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = idx / b;
    size_t t = idx % b;
    dst[idx] = src[offset + t * stride];
}

void 
globalMemStrideKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, int stride) {
    globalMemStrideKernel<<<gridDim * stride, blockDim, 0>>>(dst, src, stride, gridDim * blockDim);
}

__global__ void 
globalMemOffsetKernel(int* dst, int* src, size_t size, size_t offset)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[(idx + offset) % size];
}

void 
globalMemOffsetKernel_Wrapper(int gridDim, int blockDim, int* dst, int* src, size_t offset) {
    size_t size = size_t(gridDim) * size_t(blockDim);
	globalMemOffsetKernel<<<gridDim, blockDim, 0>>>(dst, src, size, offset);
}

