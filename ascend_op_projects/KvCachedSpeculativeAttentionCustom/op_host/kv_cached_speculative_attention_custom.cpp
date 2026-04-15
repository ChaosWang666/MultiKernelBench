
#include "kv_cached_speculative_attention_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    KvCachedSpeculativeAttentionCustomTilingData tiling;

    const gert::StorageShape* qShape = context->GetInputShape(0);
    const gert::StorageShape* kShape = context->GetInputShape(1);

    uint32_t batchSize = qShape->GetStorageShape().GetDim(0);
    uint32_t qLen = qShape->GetStorageShape().GetDim(1);
    uint32_t dModel = qShape->GetStorageShape().GetDim(2);
    uint32_t kvLen = kShape->GetStorageShape().GetDim(1);

    float scale = 1.0f / std::sqrt((float)dModel);

    tiling.set_batchSize(batchSize);
    tiling.set_qLen(qLen);
    tiling.set_kvLen(kvLen);
    tiling.set_dModel(dModel);
    tiling.set_scale(scale);

    // Use batchSize * qLen as block dim, each block handles one query vector
    uint32_t blockDim = batchSize * qLen;
    if (blockDim > 32) blockDim = 32;
    context->SetBlockDim(blockDim);

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
    const gert::Shape* qShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *qShape;
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
class KvCachedSpeculativeAttentionCustom : public OpDef {
public:
    explicit KvCachedSpeculativeAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_cache")
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

OP_ADD(KvCachedSpeculativeAttentionCustom);
}
