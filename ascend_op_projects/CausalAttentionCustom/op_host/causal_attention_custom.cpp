
#include "causal_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CausalAttentionCustomTilingData tiling;
    
    // Q shape: (batch * numHeads, seqLen, headDim)
    const gert::Shape* q_shape = context->GetInputShape(0);
    uint32_t bnh = q_shape->GetDim(0);   // batch * numHeads
    uint32_t seqLen = q_shape->GetDim(1);
    uint32_t headDim = q_shape->GetDim(2);
    
    // We set numHeads = 1 here; bnh encodes batch*numHeads
    uint32_t numHeads = 1;
    uint32_t batchSize = bnh;
    
    float scale = 1.0f / std::sqrt((float)headDim);
    
    // Each block processes one (batch*head) instance
    uint32_t blockDim = bnh;
    if (blockDim > 32) blockDim = 32;
    
    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_headDim(headDim);
    tiling.set_numHeads(numHeads);
    tiling.set_scale(scale);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // Workspace for attention scores: each block needs seqLen * seqLen floats
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = bnh * seqLen * seqLen * sizeof(float);
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
class CausalAttentionCustom : public OpDef {
public:
    explicit CausalAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(CausalAttentionCustom);
}
