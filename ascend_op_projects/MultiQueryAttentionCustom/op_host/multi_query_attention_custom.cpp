
#include "multi_query_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MultiQueryAttentionCustomTilingData tiling;

    auto attrs = context->GetAttrs();
    uint32_t batchSize = *(attrs->GetInt(0));
    uint32_t seqLen = *(attrs->GetInt(1));
    uint32_t numHeads = *(attrs->GetInt(2));
    uint32_t headDim = *(attrs->GetInt(3));

    uint32_t totalBlocks = batchSize * numHeads;
    context->SetBlockDim(totalBlocks);

    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_numHeads(numHeads);
    tiling.set_headDim(headDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // Workspace for attention scores: seqLen * seqLen per block
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = totalBlocks * seqLen * seqLen * sizeof(float);
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
class MultiQueryAttentionCustom : public OpDef {
public:
    explicit MultiQueryAttentionCustom(const char* name) : OpDef(name)
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

        this->Attr("batch_size").Int();
        this->Attr("seq_len").Int();
        this->Attr("num_heads").Int();
        this->Attr("head_dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultiQueryAttentionCustom);
}
