
#include "sparse_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SparseAttentionCustomTilingData tiling;

    auto attrs = context->GetAttrs();
    uint32_t batchSize = *(attrs->GetAttrPointer<int32_t>(0));
    uint32_t numHeads = *(attrs->GetAttrPointer<int32_t>(1));
    uint32_t seqLen = *(attrs->GetAttrPointer<int32_t>(2));
    uint32_t headDim = *(attrs->GetAttrPointer<int32_t>(3));
    uint32_t windowSize = *(attrs->GetAttrPointer<int32_t>(4));

    uint32_t totalBatchHeads = batchSize * numHeads;
    uint32_t blockDim = totalBatchHeads;
    if (blockDim > 32) blockDim = 32;

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_numHeads(numHeads);
    tiling.set_seqLen(seqLen);
    tiling.set_headDim(headDim);
    tiling.set_windowSize(windowSize);
    tiling.set_totalBatchHeads(totalBatchHeads);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = totalBatchHeads * seqLen * seqLen * sizeof(float);
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
class SparseAttentionCustom : public OpDef {
public:
    explicit SparseAttentionCustom(const char* name) : OpDef(name)
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
        this->Attr("batchSize").AttrType(REQUIRED).Int();
        this->Attr("numHeads").AttrType(REQUIRED).Int();
        this->Attr("seqLen").AttrType(REQUIRED).Int();
        this->Attr("headDim").AttrType(REQUIRED).Int();
        this->Attr("windowSize").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SparseAttentionCustom);
}
