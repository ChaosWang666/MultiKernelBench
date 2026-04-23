
#include "conv3d_max_log_sum_exp_relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t TILE_LEN  = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMaxLogSumExpReluCustomTilingData tiling;
    const gert::Shape xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t dimNum = xShape.GetDimNum();
    uint32_t B = xShape.GetDim(0);
    uint32_t C = xShape.GetDim(1);
    uint32_t spatial = 1;
    for (uint32_t i = 2; i < dimNum; i++) {
        spatial *= xShape.GetDim(i);
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(B);
    tiling.set_channels(C);
    tiling.set_spatial(spatial);
    tiling.set_tileLen(TILE_LEN);

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
    int dimNum = x_shape->GetDimNum();
    y_shape->SetDimNum(dimNum);
    for (int i = 0; i < dimNum; i++) {
        if (i == 1) {
            y_shape->SetDim(i, 1);
        } else {
            y_shape->SetDim(i, x_shape->GetDim(i));
        }
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
class Conv3dMaxLogSumExpReluCustom : public OpDef {
public:
    explicit Conv3dMaxLogSumExpReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(Conv3dMaxLogSumExpReluCustom);
}
