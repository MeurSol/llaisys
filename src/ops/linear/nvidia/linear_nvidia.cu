#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void linear_kernel(T *out, const T *in, const T *weight, const T *bias,
                              size_t batch_size, size_t in_features, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = batch_size * out_features;
    if (idx >= numel) {
        return;
    }

    size_t row = idx / out_features;
    size_t col = idx % out_features;

    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        float sum = 0.0f;
        for (size_t k = 0; k < in_features; ++k) {
            sum += utils::cast_device<float>(in[row * in_features + k]) *
                   utils::cast_device<float>(weight[col * in_features + k]);
        }
        if (bias != nullptr) {
            sum += utils::cast_device<float>(bias[col]);
        }
        out[idx] = utils::cast_device<T>(sum);
    } else {
        double sum = 0.0;
        for (size_t k = 0; k < in_features; ++k) {
            sum += static_cast<double>(in[row * in_features + k]) *
                   static_cast<double>(weight[col * in_features + k]);
        }
        if (bias != nullptr) {
            sum += static_cast<double>(bias[col]);
        }
        out[idx] = static_cast<T>(sum);
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch_size, size_t in_features, size_t out_features) {
    const int block_size = 256;
    const size_t numel = batch_size * out_features;
    const int num_blocks = static_cast<int>((numel + block_size - 1) / block_size);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_F16:
        linear_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            batch_size, in_features, out_features);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_kernel<<<num_blocks, block_size>>>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            batch_size, in_features, out_features);
        break;
    default:
        std::fprintf(stderr, "[ERROR] Unsupported data type for CUDA linear: %d\n", type);
        throw std::runtime_error("Unsupported data type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA ERROR] linear kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

} // namespace llaisys::ops::nvidia
