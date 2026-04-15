
#include "convtranspose2d_softmax_biasadd_scaling_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose2dSoftmaxBiasaddScalingSigmoidCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t channels = xShape->GetStorageShape().GetDim(1);
    uint32_t height = xShape->GetStorageShape().GetDim(2);
    uint32_t width = xShape->GetStorageShape().GetDim(3);
    uint32_t totalLength = batchSize * channels * height * width;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_scalingFactor(2.0f);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom : public OpDef {
public:
    explicit Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom);
}
