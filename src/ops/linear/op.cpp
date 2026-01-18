#include "op.hpp"
#include <cassert>
#include "cpu/linear_cpu.hpp"
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    if (bias) CHECK_SAME_DTYPE(in->dtype(), bias->dtype());

    ASSERT(in->isContiguous() && weight->isContiguous(), "Linear: input tensor and weight tensor must be contiguous.");
    
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), in->dtype(), in->shape()[0], in->shape()[1], weight->shape()[0]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
