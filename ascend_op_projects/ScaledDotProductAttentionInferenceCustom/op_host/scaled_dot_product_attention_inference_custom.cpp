
#include "scaled_dot_product_attention_inference_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaledDotProductAttentionInferenceCustomTilingData tiling;

    auto qShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = qShape.GetDim(0);
    uint32_t seqLen = qShape.GetDim(1);
    uint32_t dModel = qShape.GetDim(2);

    // We'll process one row of the attention matrix at a time per tile
    // tileSeqLen controls how many query rows we process per tile iteration
    uint32_t tileSeqLen = 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_dModel(dModel);
    tiling.set_tileSeqLen(tileSeqLen);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace: we need space for scores (seqLen floats) per query row processing
    // Plus space for intermediate results
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = batchSize * seqLen * seqLen * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    gert::Shape* o_shape = context->GetOutputShape(0);
    *o_shape = *q_shape;
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
class ScaledDotProductAttentionInferenceCustom : public OpDef {
public:
    explicit ScaledDotProductAttentionInferenceCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o")
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

OP_ADD(ScaledDotProductAttentionInferenceCustom);
}
