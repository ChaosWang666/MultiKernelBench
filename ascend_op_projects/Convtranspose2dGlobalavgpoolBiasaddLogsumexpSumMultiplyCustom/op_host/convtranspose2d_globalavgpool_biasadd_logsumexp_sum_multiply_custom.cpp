
#include "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    uint32_t batchSize = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t height = inputShape->GetOriginShape().GetDim(2);
    uint32_t width = inputShape->GetOriginShape().GetDim(3);
    uint32_t outChannels = 1; // Assuming output channel after global avg pool
    uint32_t kernelSize = 3; // Default kernel size
    uint32_t totalElements = batchSize * outChannels * 1 * 1; // Simplified calculation
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_totalElements(totalElements);
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
    // Output shape: [batchSize, 1, 1, 1] -> flattened to [batchSize]
    y_shape->SetDim(0, x1_shape->GetOriginShape().GetDim(0));
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, 1);
    y_shape->SetDim(3, 1);
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
class Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustom : public OpDef {
public:
    explicit Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
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

OP_ADD(Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustom);
}
