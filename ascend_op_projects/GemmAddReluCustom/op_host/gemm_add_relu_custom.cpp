
#include "gemm_add_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmAddReluCustomTilingData tilingData;

    uint32_t M = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t K = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t N = context->GetInputShape(1)->GetOriginShape().GetDim(0);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    matmul_tiling::MatmulApiTiling mmTiling(ascendcPlatform);
    mmTiling.SetAType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT, false);
    mmTiling.SetBType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT, true);
    mmTiling.SetCType(matmul_tiling::TPosition::GM,
                      matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetBiasType(matmul_tiling::TPosition::GM,
                         matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT);
    mmTiling.SetShape(M, N, K);
    mmTiling.SetOrgShape(M, N, K);
    mmTiling.SetBias(true);
    mmTiling.SetBufferSpace(-1, -1, -1);

    if (mmTiling.GetTiling(tilingData.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    uint32_t singleCoreM = tilingData.cubeTilingData.get_singleCoreM();
    uint32_t singleCoreN = tilingData.cubeTilingData.get_singleCoreN();
    uint32_t mCnt = (M + singleCoreM - 1) / singleCoreM;
    uint32_t nCnt = (N + singleCoreN - 1) / singleCoreN;
    uint32_t usedBlockDim = mCnt * nCnt;
    if (usedBlockDim == 0) usedBlockDim = 1;

    context->SetBlockDim(usedBlockDim);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                             context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

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
class GemmAddReluCustom : public OpDef {
public:
    explicit GemmAddReluCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmAddReluCustom);
}
