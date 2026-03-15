#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    // Always support CPU calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            out->shape(),
            out->strides(),
            in->strides());
    }

#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        ASSERT(out->isContiguous() && in->isContiguous(),
               "Rearrange CUDA: only contiguous tensors are supported for now.");
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
        auto &runtime = llaisys::core::context().runtime();
        runtime.api()->memcpy_sync(
            out->data(),
            in->data(),
            out->numel() * out->elementSize(),
            LLAISYS_MEMCPY_D2D);
        return;
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
