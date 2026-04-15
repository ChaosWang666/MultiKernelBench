
#include "conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_inChannels(context->GetAttrInt("in_channels"));
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));
    tiling.set_scale1(context->GetAttrFloat("scale1"));
    tiling.set_scale2(context->GetAttrFloat("scale2"));
    tiling.set_biasShape0(context->GetAttrInt("bias_shape")[0]);
    tiling.set_biasShape1(context->GetAttrInt("bias_shape")[1]);
    tiling.set_biasShape2(context->GetAttrInt("bias_shape")[2]);
    tiling.set_biasShape3(context->GetAttrInt("bias_shape")[3]);
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
    gert::Shape* output_shape = context->GetOutputShape(0);
    *output_shape = *input_shape;
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
class ConvTranspose3dScalingAvgPoolBiasAddScalingCustom : public OpDef {
public:
    explicit ConvTranspose3dScalingAvgPoolBiasAddScalingCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("in_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("kernel_size").SetType(INT).SetParamType(REQUIRED);
        this->Attr("stride").SetType(INT).SetParamType(REQUIRED);
        this->Attr("padding").SetType(INT).SetParamType(REQUIRED);
        this->Attr("scale1").SetType(FLOAT).SetParamType(REQUIRED);
        this->Attr("scale2").SetType(FLOAT).SetParamType(REQUIRED);
        this->Attr("bias_shape").SetType(INT_ARRAY).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dScalingAvgPoolBiasAddScalingCustom);
}
