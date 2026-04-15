
#include "conv_depthwise_2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dSquareInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* output_shape = context->GetOutputShape(0);
    uint32_t channel = weight_shape->GetOriginShape().GetDim(0);
    uint32_t kernelH = weight_shape->GetOriginShape().GetDim(2);
    uint32_t kernelW = weight_shape->GetOriginShape().GetDim(3);
    uint32_t strideH = context->GetAttrInt("stride_h");
    uint32_t strideW = context->GetAttrInt("stride_w");
    uint32_t padH = context->GetAttrInt("pad_h");
    uint32_t padW = context->GetAttrInt("pad_w");
    uint32_t dilationH = context->GetAttrInt("dilation_h");
    uint32_t dilationW = context->GetAttrInt("dilation_w");
    uint32_t inputH = input_shape->GetOriginShape().GetDim(2);
    uint32_t inputW = input_shape->GetOriginShape().GetDim(3);
    uint32_t outputH = output_shape->GetOriginShape().GetDim(2);
    uint32_t outputW = output_shape->GetOriginShape().GetDim(3);
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_channel(channel);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
    tiling.set_inputH(inputH);
    tiling.set_inputW(inputW);
    tiling.set_outputH(outputH);
    tiling.set_outputW(outputW);
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
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    y_shape->GetOriginShape().SetDim(1, w_shape->GetOriginShape().GetDim(0));
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
class ConvDepthwise2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OHWI});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Attr("stride_h").SetType(ATTR_TYPE_INT).SetDefault(1);
        this->Attr("stride_w").SetType(ATTR_TYPE_INT).SetDefault(1);
        this->Attr("pad_h").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("pad_w").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("dilation_h").SetType(ATTR_TYPE_INT).SetDefault(1);
        this->Attr("dilation_w").SetType(ATTR_TYPE_INT).SetDefault(1);
        this->Attr("channel").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("kernel_h").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("kernel_w").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("input_h").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("input_w").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("output_h").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("output_w").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvDepthwise2dSquareInputAsymmetricKernelCustom);
}
