
#include "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetOriginShape();
    // shape: [N, H, W, C]
    uint32_t N = shape.GetDim(0);
    uint32_t H = shape.GetDim(1);
    uint32_t W = shape.GetDim(2);
    uint32_t totalPositions = N * H * W;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalPositions(totalPositions);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    int64_t dimNum = y_shape->GetDimNum();
    if (dimNum > 0) {
        y_shape->SetDim(dimNum - 1, 1);
    }
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
class Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom : public OpDef {
public:
    explicit Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
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

OP_ADD(Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom);
}
