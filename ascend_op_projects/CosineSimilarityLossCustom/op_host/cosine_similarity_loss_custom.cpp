
#include "cosine_similarity_loss_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CosineSimilarityLossCustomTilingData tiling;
    const gert::StorageShape* predShape = context->GetInputShape(0);
    uint32_t batchSize = predShape->GetStorageShape().GetDim(0);
    uint32_t featureSize = predShape->GetStorageShape().GetDim(1);

    // Use batch_size as block dim, but cap at 32
    uint32_t blockDim = batchSize < 32 ? batchSize : 32;
    context->SetBlockDim(blockDim);

    tiling.set_batchSize(batchSize);
    tiling.set_featureSize(featureSize);
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
    // Output is a scalar (shape [1])
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = gert::Shape({1});
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
class CosineSimilarityLossCustom : public OpDef {
public:
    explicit CosineSimilarityLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
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

OP_ADD(CosineSimilarityLossCustom);
}
