
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor conv2d_min_add_multiply_custom_impl_npu(const at::Tensor& x, const at::Tensor& bias) {
    at::Tensor result = at::empty_like(x);
    EXEC_NPU_CMD(aclnnConv2dMinAddMultiplyCustom, x, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv2d_min_add_multiply_custom", &conv2d_min_add_multiply_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_min_add_multiply_custom", &conv2d_min_add_multiply_custom_impl_npu, "fused min + add_bias + multiply");
}
