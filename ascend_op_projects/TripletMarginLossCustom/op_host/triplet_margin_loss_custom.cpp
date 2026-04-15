
#include "triplet_margin_loss_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TripletMarginLossCustomTilingData tiling;
    const gert::StorageShape* anchorShape = context->GetInputShape(0);
    uint32_t batchSize = anchorShape->GetStorageShape().GetDim(0);
    uint32_t dimSize = anchorShape->GetStorageShape().GetDim(1);

    // Use enough blocks but not more than batch size
    uint32_t blockDim = 32;
    if (blockDim > batchSize) blockDim = batchSize;

    context->SetBlockDim(blockDim);

    tiling.set_batchSize(batchSize);
    tiling.set_dimSize(dimSize);

    // Get margin from attrs - default 1.0
    const float* marginPtr = context->GetAttrs()->GetAttrPointer<float>(0);
    float margin = 1.0f;
    if (marginPtr != nullptr) {
        margin = *marginPtr;
    }
    tiling.set_margin(margin);

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
    gert::Shape* y_shape = context->GetOutputShape(0);
    // Output is a scalar (shape [1])
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
class TripletMarginLossCustom : public OpDef {
public:
    explicit TripletMarginLossCustom(const char* name) : OpDef(name)
    {
        this->Input("anchor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positive")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("negative")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("margin").AttrType(OPTIONAL).Float(1.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TripletMarginLossCustom);
}
