
#include "multi_head_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MultiHeadAttentionCustomTilingData tiling;
    
    const gert::Shape* queryShape = context->GetInputShape(0);
    uint32_t batchSize = queryShape->GetDim(0);
    uint32_t seqLen = queryShape->GetDim(1);
    uint32_t dModel = queryShape->GetDim(2);
    
    // Infer numHeads from qkv_weight shape: [3*dModel, dModel]
    // headDim = dModel / numHeads
    // We pass numHeads via attrs or deduce. Let's use a fixed approach:
    // qkv_weight shape is [3*dModel, dModel], out_weight is [dModel, dModel]
    // We'll set numHeads = 8 as default, or deduce from shapes
    uint32_t numHeads = 8;
    uint32_t headDim = dModel / numHeads;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_dModel(dModel);
    tiling.set_numHeads(numHeads);
    tiling.set_headDim(headDim);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // Workspace for intermediate results
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = batchSize * seqLen * dModel * 3 * sizeof(float) + 
                          batchSize * numHeads * seqLen * seqLen * sizeof(float) +
                          batchSize * seqLen * dModel * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *queryShape;
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
class MultiHeadAttentionCustom : public OpDef {
public:
    explicit MultiHeadAttentionCustom(const char* name) : OpDef(name)
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
        this->Input("qkv_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("qkv_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_bias")
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MultiHeadAttentionCustom);
}
