
#include "embedding_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    EmbeddingCustomTilingData tiling;
    
    const gert::Shape* indicesShape = context->GetInputShape(1);
    uint32_t numIndices = indicesShape->GetOriginShape().GetShapeSize();
    
    const gert::Shape* weightShape = context->GetInputShape(0);
    uint32_t embeddingDim = weightShape->GetOriginShape().GetDim(1);
    
    uint32_t blockDim = 32;
    context->SetBlockDim(blockDim);
    
    tiling.set_numIndices(numIndices);
    tiling.set_embeddingDim(embeddingDim);
    
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
    const gert::Shape* weightShape = context->GetInputShape(0);
    const gert::Shape* indicesShape = context->GetInputShape(1);
    gert::Shape* outShape = context->GetOutputShape(0);
    
    // Output shape = indices_shape + [embedding_dim]
    int indicesDimNum = indicesShape->GetOriginShape().GetDimNum();
    int64_t embDim = weightShape->GetOriginShape().GetDim(1);
    
    std::vector<int64_t> outDims;
    for (int i = 0; i < indicesDimNum; i++) {
        outDims.push_back(indicesShape->GetOriginShape().GetDim(i));
    }
    outDims.push_back(embDim);
    
    *outShape = gert::Shape(outDims.data(), outDims.size());
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
class EmbeddingCustom : public OpDef {
public:
    explicit EmbeddingCustom(const char* name) : OpDef(name)
    {
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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

OP_ADD(EmbeddingCustom);
}
