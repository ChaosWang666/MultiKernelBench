
#include "nearest_neighbor_upsample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    NearestNeighborUpsampleCustomTilingData tiling;
    
    const gert::Shape* inputShape = context->GetInputShape(0);
    uint32_t batchSize = inputShape->GetDim(0);
    uint32_t channels = inputShape->GetDim(1);
    uint32_t inputHeight = inputShape->GetDim(2);
    uint32_t inputWidth = inputShape->GetDim(3);
    uint32_t scaleFactor = 4;
    uint32_t outputHeight = inputHeight * scaleFactor;
    uint32_t outputWidth = inputWidth * scaleFactor;
    uint32_t totalOutputElements = batchSize * channels * outputHeight * outputWidth;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_scaleFactor(scaleFactor);
    tiling.set_totalOutputElements(totalOutputElements);
    
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
    // NCHW: output is N, C, H*4, W*4
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, x_shape->GetDim(2) * 4);
    y_shape->SetDim(3, x_shape->GetDim(3) * 4);
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
class NearestNeighborUpsampleCustom : public OpDef {
public:
    explicit NearestNeighborUpsampleCustom(const char* name) : OpDef(name)
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

OP_ADD(NearestNeighborUpsampleCustom);
}
