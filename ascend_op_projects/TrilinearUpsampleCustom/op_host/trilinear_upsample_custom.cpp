
#include "trilinear_upsample_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TrilinearUpsampleCustomTilingData tiling;
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint32_t batchSize = inputShape->GetStorageShape().GetDim(0);
    uint32_t channels = inputShape->GetStorageShape().GetDim(1);
    uint32_t inputDepth = inputShape->GetStorageShape().GetDim(2);
    uint32_t inputHeight = inputShape->GetStorageShape().GetDim(3);
    uint32_t inputWidth = inputShape->GetStorageShape().GetDim(4);
    uint32_t outputDepth = inputDepth * 2;
    uint32_t outputHeight = inputHeight * 2;
    uint32_t outputWidth = inputWidth * 2;

    uint32_t totalBC = batchSize * channels;
    uint32_t blockDim = totalBC < 32 ? totalBC : 32;

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_inputDepth(inputDepth);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_outputDepth(outputDepth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
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
    // Output shape: [N, C, D*2, H*2, W*2]
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, x_shape->GetDim(2) * 2);
    y_shape->SetDim(3, x_shape->GetDim(3) * 2);
    y_shape->SetDim(4, x_shape->GetDim(4) * 2);
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
class TrilinearUpsampleCustom : public OpDef {
public:
    explicit TrilinearUpsampleCustom(const char* name) : OpDef(name)
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

OP_ADD(TrilinearUpsampleCustom);
}
