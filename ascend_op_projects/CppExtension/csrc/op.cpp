
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor conv3d_min_softmax_custom_impl_npu(const at::Tensor& self) {
    auto sizes = self.sizes();
    std::vector<int64_t> outSize = {sizes[0], sizes[1], sizes[3], sizes[4]};
    at::Tensor result = at::empty(outSize, self.options());
    EXEC_NPU_CMD(aclnnConv3dMinSoftmaxCustom, self, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("conv3d_min_softmax_custom", &conv3d_min_softmax_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_min_softmax_custom", &conv3d_min_softmax_custom_impl_npu, "conv3d + min(dim=2) + softmax(dim=1)");
}
