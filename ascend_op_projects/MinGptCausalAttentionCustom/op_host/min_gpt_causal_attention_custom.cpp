
#include "min_gpt_causal_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MinGptCausalAttentionCustomTilingData tiling;
    uint32_t batch = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t seqLen = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t nEmb = context->GetAttrInt64("n_embd");
    uint32_t nHead = context->GetAttrInt64("n_head");
    uint32_t maxSeqlen = context->GetAttrInt64("max_seqlen");
    float attnPdrop = context->GetAttrFloat("attn_pdrop");
    float residPdrop = context->GetAttrFloat("resid_pdrop");

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_seqLen(seqLen);
    tiling.set_nEmb(nEmb);
    tiling.set_nHead(nHead);
    tiling.set_maxSeqlen(maxSeqlen);
    tiling.set_attnPdrop(attnPdrop);
    tiling.set_residPdrop(residPdrop);
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
class MinGptCausalAttentionCustom : public OpDef {
public:
    explicit MinGptCausalAttentionCustom(const char* name) : OpDef(name)
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
        this->Attr("n_embd").SetType(ATTR_TYPE_INT64).SetDefault(768);
        this->Attr("n_head").SetType(ATTR_TYPE_INT64).SetDefault(8);
        this->Attr("attn_pdrop").SetType(ATTR_TYPE_FLOAT).SetDefault(0.0f);
        this->Attr("resid_pdrop").SetType(ATTR_TYPE_FLOAT).SetDefault(0.0f);
        this->Attr("max_seqlen").SetType(ATTR_TYPE_INT64).SetDefault(1024);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MinGptCausalAttentionCustom);
}
