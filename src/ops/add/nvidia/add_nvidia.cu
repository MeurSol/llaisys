#include "add_nvidia.cuh"
#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <cstdio>

namespace llaisys::ops::nvidia {

// CUDA kernel for element-wise addition
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[idx] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(a[idx]) + llaisys::utils::cast<float>(b[idx]));
        } else {
            c[idx] = a[idx] + b[idx];
        }
    }
}

// Specialization for half using CUDA native half arithmetic
template <>
__global__ void add_kernel<llaisys::fp16_t>(llaisys::fp16_t *c, const llaisys::fp16_t *a, const llaisys::fp16_t *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Use native half2 arithmetic for better performance
        half ha = __half_raw(a[idx]._v);
        half hb = __half_raw(b[idx]._v);
        half hc = __hadd(ha, hb);
        c[idx]._v = __half_raw(hc).x;
    }
}

// Specialization for bfloat16
template <>
__global__ void add_kernel<llaisys::bf16_t>(llaisys::bf16_t *c, const llaisys::bf16_t *a, const llaisys::bf16_t *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // bfloat16: convert to float, add, convert back
        float fa = llaisys::utils::cast<float>(a[idx]);
        float fb = llaisys::utils::cast<float>(b[idx]);
        c[idx] = llaisys::utils::cast<llaisys::bf16_t>(fa + fb);
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    const int block_size = 256;
    const int num_blocks = (numel + block_size - 1) / block_size;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<llaisys::bf16_t *>(c),
            reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b),
            numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<llaisys::fp16_t *>(c),
            reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b),
            numel);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA add: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] add kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
