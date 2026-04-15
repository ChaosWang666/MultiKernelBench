
#include "gemm_divide_sum_scaling_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 1024;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    GemmDivideSumScalingCustomTilingData tiling;
    uint32_t batch_size = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t input_size = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t hidden_size = context->GetInputShape(1)->GetOriginShape().GetDim(0);
    float scaling_factor = context->GetAttrFloat("scaling_factor");
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch_size(batch_size);
    tiling.set_input_size(input_size);
    tiling.set_hidden_size(hidden_size);
    tiling.set_scaling_factor(scaling_factor);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDim(0, x1_shape->GetOriginShape().GetDim(0));
    y_shape->SetDim(1, weight_shape->GetOriginShape().GetDim(0));
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
class GemmDivideSumScalingCustom : public OpDef {
public:
    explicit GemmDivideSumScalingCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scaling_factor").SetType(ATTR_TYPE_FLOAT).SetDefault(1.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(GemmDivideSumScalingCustom);
}
