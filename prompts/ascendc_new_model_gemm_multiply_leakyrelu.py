project_json_src='''
[
    {
        "op": "GemmMultiplyLeakyreluCustom",
        "language": "cpp",
        "input_desc": [
            {"name": "x",      "param_type": "required", "format": ["ND"], "type": ["float"]},
            {"name": "weight", "param_type": "required", "format": ["ND"], "type": ["float"]},
            {"name": "bias",   "param_type": "required", "format": ["ND"], "type": ["float"]}
        ],
        "output_desc": [
            {"name": "y", "param_type": "required", "format": ["ND"], "type": ["float"]}
        ],
        "attr": [
            {"name": "multiplier",     "param_type": "required", "type": "float"},
            {"name": "negative_slope", "param_type": "required", "type": "float"}
        ]
    }
]
'''

host_tiling_src="""
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmMultiplyLeakyreluCustomTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, mPerAic);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
  TILING_DATA_FIELD_DEF(uint32_t, aicCoreNum);
  TILING_DATA_FIELD_DEF(float,    multiplier);
  TILING_DATA_FIELD_DEF(float,    negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmMultiplyLeakyreluCustom, GemmMultiplyLeakyreluCustomTilingData)
}
"""

host_operator_src="""
#include "gemm_multiply_leakyrelu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    GemmMultiplyLeakyreluCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    auto wShape = context->GetInputShape(1)->GetOriginShape();
    uint32_t M = xShape.GetDim(0);
    uint32_t K = xShape.GetDim(1);
    uint32_t N = wShape.GetDim(0);

    uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    if (aicCoreNum == 0) { aicCoreNum = 1; }
    uint32_t mPerAic = (M + aicCoreNum - 1) / aicCoreNum;
    if (mPerAic == 0) { mPerAic = 1; }
    if (mPerAic % 16 != 0) { mPerAic = ((mPerAic + 15) / 16) * 16; }
    uint32_t actualUsed = (M + mPerAic - 1) / mPerAic;
    if (actualUsed == 0) { actualUsed = 1; }

    matmul_tiling::MatmulApiTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetShape(mPerAic, N, K);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    if (cubeTiling.GetTiling(tiling.cubeTiling) == -1) { return ge::GRAPH_FAILED; }

    auto* attrs = context->GetAttrs();
    float multiplier     = *attrs->GetAttrPointer<float>(0);
    float negativeSlope  = *attrs->GetAttrPointer<float>(1);

    tiling.set_M(M); tiling.set_N(N); tiling.set_K(K);
    tiling.set_mPerAic(mPerAic);
    tiling.set_tileLen(8192);
    tiling.set_aicCoreNum(actualUsed);
    tiling.set_multiplier(multiplier);
    tiling.set_negativeSlope(negativeSlope);

    context->SetBlockDim(actualUsed);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, wShape->GetDim(0));
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GemmMultiplyLeakyreluCustom : public OpDef {
public:
    explicit GemmMultiplyLeakyreluCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("multiplier").AttrType(REQUIRED).Float();
        this->Attr("negative_slope").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(GemmMultiplyLeakyreluCustom);
}
"""

kernel_src="""
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

constexpr int32_t BUFFER_NUM = 2;

extern "C" __global__ __aicore__ void gemm_multiply_leakyrelu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);
    AscendC::TPipe pipe;

    uint32_t M = td.M, N = td.N;
    uint32_t mPerAic = td.mPerAic, aicCoreNum = td.aicCoreNum, tileLen = td.tileLen;
    float multiplier = td.multiplier;
    float negativeSlope = td.negativeSlope;

    if ASCEND_IS_AIC {
        uint32_t aicId = AscendC::GetBlockIdx();
        if (aicId >= aicCoreNum) return;
        uint32_t mStart = aicId * mPerAic;
        if (mStart >= M) return;
        uint32_t curM = (mStart + mPerAic > M) ? (M - mStart) : mPerAic;

        AscendC::GlobalTensor<float> xGm, wGm, biasGm, yGm;
        xGm.SetGlobalBuffer((__gm__ float*)x + mStart * td.K, curM * td.K);
        wGm.SetGlobalBuffer((__gm__ float*)weight, N * td.K);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, N);
        yGm.SetGlobalBuffer((__gm__ float*)y + mStart * N, curM * N);

        using AscendC::TPosition; using AscendC::CubeFormat; using AscendC::MatmulType;
        typedef MatmulType<TPosition::GM, CubeFormat::ND, float> aType;
        typedef MatmulType<TPosition::GM, CubeFormat::ND, float> bType;
        typedef MatmulType<TPosition::GM, CubeFormat::ND, float> cType;
        typedef MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;

        AscendC::Matmul<aType, bType, cType, biasType> mm;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &td.cubeTiling);
        mm.SetTensorA(xGm);
        mm.SetTensorB(wGm, true);
        mm.SetBias(biasGm);
        mm.SetTail(curM, N, td.K);
        mm.IterateAll(yGm);
        mm.End();

        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0);
    }

    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(0);
        uint32_t aicId = AscendC::GetBlockIdx();
        uint32_t subId = AscendC::GetSubBlockIdx();
        if (aicId >= aicCoreNum) return;
        uint32_t mStart = aicId * mPerAic;
        if (mStart >= M) return;
        uint32_t curM = (mStart + mPerAic > M) ? (M - mStart) : mPerAic;

        uint32_t halfM = (curM + 1) / 2;
        uint32_t aivMStart = mStart + subId * halfM;
        uint32_t aivMEnd = aivMStart + halfM;
        if (aivMStart >= mStart + curM) return;
        if (aivMEnd > mStart + curM) aivMEnd = mStart + curM;
        uint32_t aivM = aivMEnd - aivMStart;

        AscendC::GlobalTensor<float> yGm;
        yGm.SetGlobalBuffer((__gm__ float*)y + aivMStart * N, aivM * N);

        uint32_t totalLen = aivM * N;
        AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qIn;
        AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> qOut;
        pipe.InitBuffer(qIn,  BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(qOut, BUFFER_NUM, tileLen * sizeof(float));

        uint32_t tileNum = (totalLen + tileLen - 1) / tileLen;
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t off = i * tileLen;
            uint32_t cur = (off + tileLen > totalLen) ? (totalLen - off) : tileLen;

            AscendC::LocalTensor<float> in = qIn.AllocTensor<float>();
            AscendC::DataCopy(in, yGm[off], cur);
            qIn.EnQue(in);

            AscendC::LocalTensor<float> in2 = qIn.DeQue<float>();
            AscendC::LocalTensor<float> out = qOut.AllocTensor<float>();
            // y = LeakyRelu(gemm(x, W^T) * multiplier, negative_slope)
            AscendC::Muls(out, in2, multiplier, cur);
            AscendC::LeakyRelu(out, out, negativeSlope, cur);
            qIn.FreeTensor(in2);
            qOut.EnQue(out);

            AscendC::LocalTensor<float> out2 = qOut.DeQue<float>();
            AscendC::DataCopy(yGm[off], out2, cur);
            qOut.FreeTensor(out2);
        }
    }
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

at::Tensor gemm_multiply_leakyrelu_custom_impl_npu(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& bias,
    double multiplier, double negative_slope)
{
    auto xS = x.sizes();
    auto wS = weight.sizes();
    int64_t M = xS[0], N = wS[0];
    at::Tensor result = at::empty({M, N}, x.options());
    EXEC_NPU_CMD(aclnnGemmMultiplyLeakyreluCustom, x, weight, bias,
                 multiplier, negative_slope, result);
    return result;
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("gemm_multiply_leakyrelu_custom", &gemm_multiply_leakyrelu_custom_impl_npu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_multiply_leakyrelu_custom", &gemm_multiply_leakyrelu_custom_impl_npu,
          "fused gemm + multiply + leaky_relu");
}
"""

model_src='''
import torch
import torch_npu
import custom_ops_lib

class ModelNew(torch.nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        # 用 nn.Parameter 显式声明 reference 中 nn.Linear 内部的 weight 和 bias;
        # 不允许 self.gemm = nn.Linear(...) 之类的 nn.Module 实例化。
        self.gemm_weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.gemm_bias   = torch.nn.Parameter(torch.randn(out_features))
        # 标量 init_attr 直接存到 self
        self.multiplier     = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        # forward 必须是单行 return custom_ops_lib.<op>_custom(...);
        # 参数只能是 forward 入参 / self.<name>, tensor 参数可加 .contiguous() 等。
        return custom_ops_lib.gemm_multiply_leakyrelu_custom(
            x.contiguous(),
            self.gemm_weight.contiguous(),
            self.gemm_bias.contiguous(),
            self.multiplier,
            self.negative_slope,
        )
'''
