
#include "conv2d_min_tanh_tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dMinTanhTanhCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batch = xShape->GetStorageShape().GetDim(0);
    uint32_t channels = xShape->GetStorageShape().GetDim(1);
    uint32_t height = xShape->GetStorageShape().GetDim(2);
    uint32_t width = xShape->GetStorageShape().GetDim(3);
    uint32_t spatialSize = height * width;
    uint32_t totalLength = batch * spatialSize;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_channels(channels);
    tiling.set_spatialSize(spatialSize);
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
    // Output: [batch, 1, H, W]
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, x_shape->GetDim(2));
    y_shape->SetDim(3, x_shape->GetDim(3));
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
class Conv2dMinTanhTanhCustom : public OpDef {
public:
    explicit Conv2dMinTanhTanhCustom(const char* name) : OpDef(name)
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

OP_ADD(Conv2dMinTanhTanhCustom);
}
