
#include "gemm_sigmoid_scaling_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmSigmoidScalingResidualAddCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    auto wShape = context->GetInputShape(1)->GetOriginShape();
    uint32_t M = xShape.GetDim(0);
    uint32_t K = xShape.GetDim(1);
    uint32_t N = wShape.GetDim(0);

    auto attrs = context->GetAttrs();
    float scalingFactor = *(attrs->GetFloat(0));

    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);
    tiling.set_scalingFactor(scalingFactor);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(20);
    cubeTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
    cubeTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, true);
    cubeTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(20);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 32 * 1024 * 1024;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, w_shape->GetDim(0));
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class GemmSigmoidScalingResidualAddCustom : public OpDef {
public:
    explicit GemmSigmoidScalingResidualAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("scaling_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GemmSigmoidScalingResidualAddCustom);
}
