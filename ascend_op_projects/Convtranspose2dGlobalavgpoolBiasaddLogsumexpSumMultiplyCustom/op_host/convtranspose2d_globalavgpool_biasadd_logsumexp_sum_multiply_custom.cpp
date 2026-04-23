
#include "convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustomTilingData tiling;
    const gert::Shape& xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = (uint32_t)xShape.GetDim(0);
    uint32_t channels = (uint32_t)xShape.GetDim(1);

    uint32_t blockDim = batchSize;
    if (blockDim == 0) {
        blockDim = 1;
    }

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);

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
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
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
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(Convtranspose2dGlobalavgpoolBiasaddLogsumexpSumMultiplyCustom);
}
