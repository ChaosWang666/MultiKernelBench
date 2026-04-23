
#include "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_SIZE = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData tiling;

    auto yShape = context->GetInputShape(0)->GetOriginShape();
    auto xShape = context->GetInputShape(1)->GetOriginShape();

    uint32_t batchSize = yShape.GetDim(0);
    uint32_t nLen = yShape.GetDim(1);
    uint32_t mLen = xShape.GetDim(1);

    uint32_t rowsPerBlock = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_nLen(nLen);
    tiling.set_mLen(mLen);
    tiling.set_rowsPerBlock(rowsPerBlock);
    tiling.set_tileSize(TILE_SIZE);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(1);
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = *x_shape;
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
class GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom : public OpDef {
public:
    explicit GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom);
}
