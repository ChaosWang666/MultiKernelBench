
#include "conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const gert::Shape* norm_weight_shape = context->GetInputShape(3);
    const gert::Shape* norm_bias_shape = context->GetInputShape(4);
    
    tiling.set_batch(input_shape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(input_shape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(weight_shape->GetOriginShape().GetDim(0));
    tiling.set_height(input_shape->GetOriginShape().GetDim(2));
    tiling.set_width(input_shape->GetOriginShape().GetDim(3));
    tiling.set_kernelH(weight_shape->GetOriginShape().GetDim(2));
    tiling.set_kernelW(weight_shape->GetOriginShape().GetDim(3));
    tiling.set_groups(norm_weight_shape->GetOriginShape().GetDim(0));
    tiling.set_padH(0);
    tiling.set_padW(0);
    tiling.set_strideH(1);
    tiling.set_strideW(1);
    tiling.set_dilationH(1);
    tiling.set_dilationW(1);
    
    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetOriginShape(gert::Shape({input_shape->GetOriginShape().GetDim(0),
                                         weight_shape->GetOriginShape().GetDim(0),
                                         input_shape->GetOriginShape().GetDim(2),
                                         input_shape->GetOriginShape().GetDim(3)}));
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
class Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom : public OpDef {
public:
    explicit Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("conv_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW})
            .UnknownShapeFormat({ge::FORMAT_OIHW});
        this->Input("conv_bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("group_norm_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("group_norm_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dGroupNormTanhHardSwishResidualAddLogSumExpCustom);
}
