
#include "kv_cached_chat_batch_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    KvCachedChatBatchAttentionCustomTilingData tiling;

    // Q shape: (batch_size, q_len, d_model)
    // K shape: (batch_size, kv_len, d_model)
    // V shape: (batch_size, kv_len, d_model)
    auto qShape = context->GetInputShape(0)->GetOriginShape();
    auto kShape = context->GetInputShape(1)->GetOriginShape();

    uint32_t batchSize = qShape.GetDim(0);
    uint32_t qLen = qShape.GetDim(1);
    uint32_t dModel = qShape.GetDim(2);
    uint32_t kvLen = kShape.GetDim(1);

    // Each block processes one batch element
    uint32_t blockDim = batchSize;
    context->SetBlockDim(blockDim);

    // Tile kv_len dimension for processing
    // We want tiles that fit in local memory
    // Each tile of kv needs: tileKv * dModel * sizeof(float) for K and V
    // Plus tileKv floats for scores
    // Local memory budget per buffer ~approx. We'll tile conservatively
    uint32_t kvTileNum = 16; // number of tiles along kv dimension
    uint32_t dTileNum = 8;  // number of tiles along d dimension

    tiling.set_batchSize(batchSize);
    tiling.set_qLen(qLen);
    tiling.set_kvLen(kvLen);
    tiling.set_dModel(dModel);
    tiling.set_kvTileNum(kvTileNum);
    tiling.set_dTileNum(dTileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace for intermediate results (scores, etc.)
    // Per batch: kvLen floats for scores + dModel floats for output
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = batchSize * (kvLen + dModel) * sizeof(float);

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
class KvCachedChatBatchAttentionCustom : public OpDef {
public:
    explicit KvCachedChatBatchAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
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

OP_ADD(KvCachedChatBatchAttentionCustom);
}
