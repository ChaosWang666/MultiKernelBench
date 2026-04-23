
#include "gemm_group_norm_min_bias_add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmGroupNormMinBiasAddCustomTilingData tiling;
    auto x_shape = context->GetInputShape(0)->GetOriginShape();
    auto bias_shape = context->GetInputShape(1)->GetOriginShape();
    uint32_t totalRows = x_shape.GetDim(0);
    uint32_t totalCols = bias_shape.GetDim(1);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalRows(totalRows);
    tiling.set_totalCols(totalCols);
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
    const gert::Shape* bias_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, 1);
    y_shape->SetDim(1, bias_shape->GetDim(1));
    y_shape->SetDim(2, x_shape->GetDim(0));
    y_shape->SetDim(3, 1);
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
class GemmGroupNormMinBiasAddCustom : public OpDef {
public:
    explicit GemmGroupNormMinBiasAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(GemmGroupNormMinBiasAddCustom);
}
