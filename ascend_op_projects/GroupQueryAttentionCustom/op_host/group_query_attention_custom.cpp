
#include "group_query_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GroupQueryAttentionCustomTilingData tiling;
    
    // Q shape: [B, L, H, head_dim]
    const gert::Shape* qShape = context->GetInputShape(0);
    // K shape: [B, L, H_kv, head_dim]
    const gert::Shape* kShape = context->GetInputShape(1);
    
    uint32_t B = qShape->GetDim(0);
    uint32_t L = qShape->GetDim(1);
    uint32_t H = qShape->GetDim(2);
    uint32_t headDim = qShape->GetDim(3);
    uint32_t H_kv = kShape->GetDim(2);
    uint32_t groupSize = H / H_kv;
    
    // Use B * H blocks so each block handles one (batch, head) pair
    uint32_t blockDim = B * H;
    if (blockDim > 32) blockDim = 32;
    
    context->SetBlockDim(blockDim);
    tiling.set_batchSize(B);
    tiling.set_seqLen(L);
    tiling.set_numHeads(H);
    tiling.set_numKvHeads(H_kv);
    tiling.set_headDim(headDim);
    tiling.set_groupSize(groupSize);
    
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
class GroupQueryAttentionCustom : public OpDef {
public:
    explicit GroupQueryAttentionCustom(const char* name) : OpDef(name)
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

OP_ADD(GroupQueryAttentionCustom);
}
