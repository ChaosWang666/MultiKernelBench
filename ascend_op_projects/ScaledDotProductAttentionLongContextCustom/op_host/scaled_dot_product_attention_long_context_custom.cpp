
#include "scaled_dot_product_attention_long_context_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 1;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ScaledDotProductAttentionLongContextCustomTilingData tiling;
    
    auto qShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = qShape.GetDim(0);
    uint32_t seqLen = qShape.GetDim(1);
    uint32_t dModel = qShape.GetDim(2);
    
    // Use block sizes suitable for tiled attention
    uint32_t blockQ = 32;
    uint32_t blockKV = 32;
    
    // Number of blocks = ceil(seqLen / blockQ) * batchSize
    uint32_t numBlocks = ((seqLen + blockQ - 1) / blockQ) * batchSize;
    uint32_t blockDim = numBlocks < 32 ? numBlocks : 32;
    
    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_dModel(dModel);
    tiling.set_blockQ(blockQ);
    tiling.set_blockKV(blockKV);
    
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    
    // workspace for intermediate results
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = (size_t)batchSize * seqLen * dModel * sizeof(float);
    
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
class ScaledDotProductAttentionLongContextCustom : public OpDef {
public:
    explicit ScaledDotProductAttentionLongContextCustom(const char* name) : OpDef(name)
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

OP_ADD(ScaledDotProductAttentionLongContextCustom);
}
