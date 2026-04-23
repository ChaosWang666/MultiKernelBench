
#include "conv2d_scaling_min_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dScalingMinCustomTilingData tiling;

    const gert::Shape& inShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = inShape.GetDim(0);
    uint32_t channels = inShape.GetDim(1);
    uint32_t H = inShape.GetDim(2);
    uint32_t W = inShape.GetDim(3);
    uint32_t spatialSize = H * W;

    uint32_t tileLen = 1024;
    if (spatialSize < tileLen) {
        tileLen = spatialSize;
    }

    uint32_t blockDim = 32;
    if (batchSize < blockDim) blockDim = batchSize;
    uint32_t batchesPerBlock = (batchSize + blockDim - 1) / blockDim;
    blockDim = (batchSize + batchesPerBlock - 1) / batchesPerBlock;

    const float* scalePtr = context->GetAttrs()->GetAttrPointer<float>(0);
    float scaleFactor = (scalePtr != nullptr) ? *scalePtr : 1.0f;

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_spatialSize(spatialSize);
    tiling.set_tileLen(tileLen);
    tiling.set_scaleFactor(scaleFactor);

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
class Conv2dScalingMinCustom : public OpDef {
public:
    explicit Conv2dScalingMinCustom(const char* name) : OpDef(name)
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
        this->Attr("scale_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dScalingMinCustom);
}
