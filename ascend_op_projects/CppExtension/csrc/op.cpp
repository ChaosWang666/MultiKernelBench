
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom_impl_npu(const at::Tensor& x) {
    int64_t batch_size = x.size(0);
    at::Tensor result = at::empty({batch_size, 1}, x.options());
    EXEC_NPU_CMD(aclnnGemmLogSumExpLeakyReluLeakyReluGeluGeluCustom, x, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom", &gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom", &gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom_impl_npu, "fused logsumexp + leaky_relu + gelu");
}
