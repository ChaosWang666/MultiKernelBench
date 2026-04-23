
#include "gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmLogSumExpLeakyReluLeakyReluGeluGeluCustomTilingData tiling;
    uint32_t totalRows = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t cols = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalRows(totalRows);
    tiling.set_cols(cols);
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
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
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
class GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom : public OpDef {
public:
    explicit GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmLogSumExpLeakyReluLeakyReluGeluGeluCustom);
}
