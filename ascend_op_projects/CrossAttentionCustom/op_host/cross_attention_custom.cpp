
#include "cross_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CrossAttentionCustomTilingData tiling;

    const gert::StorageShape* qShape = context->GetInputShape(0);
    uint32_t totalBatchHeads = qShape->GetStorageShape().GetDim(0);
    uint32_t seqLenQ = qShape->GetStorageShape().GetDim(1);
    uint32_t headDim = qShape->GetStorageShape().GetDim(2);

    const gert::StorageShape* kShape = context->GetInputShape(1);
    uint32_t seqLenKV = kShape->GetStorageShape().GetDim(1);

    uint32_t blockDim = totalBatchHeads < 32 ? totalBatchHeads : 32;

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(1);
    tiling.set_seqLenQ(seqLenQ);
    tiling.set_seqLenKV(seqLenKV);
    tiling.set_headDim(headDim);
    tiling.set_numHeads(1);
    tiling.set_totalBatchHeads(totalBatchHeads);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = totalBatchHeads * seqLenQ * seqLenKV * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *qShape;
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
class CrossAttentionCustom : public OpDef {
public:
    explicit CrossAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
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

OP_ADD(CrossAttentionCustom);
}
