
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor add_bias_broadcast_custom_impl_npu(const at::Tensor& self, const at::Tensor& bias) {
    at::Tensor result = at::empty_like(self);
    EXEC_NPU_CMD(aclnnAddBiasBroadcastCustom, self, bias, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("add_bias_broadcast_custom", &add_bias_broadcast_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_bias_broadcast_custom", &add_bias_broadcast_custom_impl_npu, "x + bias");
}
