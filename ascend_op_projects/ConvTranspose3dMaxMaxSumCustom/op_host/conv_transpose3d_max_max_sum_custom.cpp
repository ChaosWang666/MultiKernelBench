
#include "conv_transpose3d_max_max_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dMaxMaxSumCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t channels = xShape->GetStorageShape().GetDim(1);
    uint32_t depth = xShape->GetStorageShape().GetDim(2);
    uint32_t height = xShape->GetStorageShape().GetDim(3);
    uint32_t width = xShape->GetStorageShape().GetDim(4);
    uint32_t totalLength = batchSize * channels * depth * height * width;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
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
    // Input: [B, C, D, H, W]
    // After MaxPool3d(2): [B, C, D/2, H/2, W/2]
    // After MaxPool3d(3): [B, C, D/6, H/6, W/6]
    // After sum(dim=1, keepdim=True): [B, 1, D/6, H/6, W/6]
    int64_t B = x_shape->GetDim(0);
    int64_t D = x_shape->GetDim(2);
    int64_t H = x_shape->GetDim(3);
    int64_t W = x_shape->GetDim(4);
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, B);
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, D / 6);
    y_shape->SetDim(3, H / 6);
    y_shape->SetDim(4, W / 6);
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
class ConvTranspose3dMaxMaxSumCustom : public OpDef {
public:
    explicit ConvTranspose3dMaxMaxSumCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose3dMaxMaxSumCustom);
}
