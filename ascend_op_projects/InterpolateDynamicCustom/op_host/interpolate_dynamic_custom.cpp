
#include "interpolate_dynamic_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    InterpolateDynamicCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t channels = x_shape->GetDim(1);
    uint32_t inputH = x_shape->GetDim(2);
    uint32_t inputW = x_shape->GetDim(3);

    const auto* attrs = context->GetAttrs();
    uint32_t outputH = *(attrs->GetAttrPointer<int32_t>(0));
    uint32_t outputW = *(attrs->GetAttrPointer<int32_t>(1));

    uint32_t totalOutputRows = batchSize * channels * outputH;
    uint32_t rowsPerCore = totalOutputRows / BLOCK_DIM;
    uint32_t remainRows = totalOutputRows % BLOCK_DIM;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_inputH(inputH);
    tiling.set_inputW(inputW);
    tiling.set_outputH(outputH);
    tiling.set_outputW(outputW);
    tiling.set_totalOutputRows(totalOutputRows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_remainRows(remainRows);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    // Output shape: [N, C, target_h, target_w]
    // We need to get attrs but for now set from input
    *y_shape = *x_shape;
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
class InterpolateDynamicCustom : public OpDef {
public:
    explicit InterpolateDynamicCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("target_h")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("target_w")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(InterpolateDynamicCustom);
}
