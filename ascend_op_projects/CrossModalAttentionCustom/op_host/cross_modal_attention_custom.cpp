
#include "cross_modal_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrossModalAttentionCustomTilingData tiling;

    // Q shape: [batch*num_heads, seqLenQ, headDim]
    // K shape: [batch*num_heads, seqLenKV, headDim]
    // V shape: [batch*num_heads, seqLenKV, headDim]
    auto qShape = context->GetInputShape(0)->GetOriginShape();
    auto kShape = context->GetInputShape(1)->GetOriginShape();

    uint32_t bnHeads = qShape.GetDim(0);
    uint32_t seqLenQ = qShape.GetDim(1);
    uint32_t headDim = qShape.GetDim(2);
    uint32_t seqLenKV = kShape.GetDim(1);

    // We'll use bnHeads as blockDim so each core handles one (batch, head) pair
    uint32_t blockDim = bnHeads;
    if (blockDim > 32) blockDim = 32;

    float scale = 1.0f / std::sqrt((float)headDim);

    tiling.set_batchSize(bnHeads);
    tiling.set_numHeads(1);
    tiling.set_seqLenQ(seqLenQ);
    tiling.set_seqLenKV(seqLenKV);
    tiling.set_headDim(headDim);
    tiling.set_dModel(headDim);
    tiling.set_scale(scale);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace: each block needs seqLenQ * seqLenKV floats for attention scores
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = bnHeads * seqLenQ * seqLenKV * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* q_shape = context->GetInputShape(0);
    gert::Shape* out_shape = context->GetOutputShape(0);
    *out_shape = *q_shape;
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
class CrossModalAttentionCustom : public OpDef {
public:
    explicit CrossModalAttentionCustom(const char* name) : OpDef(name)
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
        this->Output("out")
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

OP_ADD(CrossModalAttentionCustom);
}
