
#include "conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData tiling;
    uint32_t batch = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t inChannels = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t outChannels = context->GetOutputShape(0)->GetOriginShape().GetDim(1);
    uint32_t kernelSize = context->GetAttrInt("kernel_size");
    uint32_t stride = context->GetAttrInt("stride");
    uint32_t padding = context->GetAttrInt("padding");
    uint32_t dilation = context->GetAttrInt("dilation");
    uint32_t inputLength = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    uint32_t outputLength = (inputLength - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);
    tiling.set_inputLength(inputLength);
    tiling.set_outputLength(outputLength);
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
    uint32_t batch = x1_shape->GetOriginShape().GetDim(0);
    uint32_t outChannels = weight_shape->GetOriginShape().GetDim(0);
    uint32_t inputLength = x1_shape->GetOriginShape().GetDim(2);
    uint32_t kernelSize = weight_shape->GetOriginShape().GetDim(2);
    uint32_t stride = context->GetAttrInt("stride");
    uint32_t padding = context->GetAttrInt("padding");
    uint32_t dilation = context->GetAttrInt("dilation");
    uint32_t outputLength = (inputLength - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;
    y_shape->SetDim(0, batch);
    y_shape->SetDim(1, outChannels);
    y_shape->SetDim(2, outputLength);
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
class ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom : public OpDef {
public:
    explicit ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom(const char* name) : OpDef(name)
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
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("stride").SetType(ATTR_VALUE_INT).SetInt(1);
        this->Attr("padding").SetType(ATTR_VALUE_INT).SetInt(0);
        this->Attr("dilation").SetType(ATTR_VALUE_INT).SetInt(1);
        this->Attr("in_channels").SetType(ATTR_VALUE_INT).SetInt(1);
        this->Attr("out_channels").SetType(ATTR_VALUE_INT).SetInt(1);
        this->Attr("kernel_size").SetType(ATTR_VALUE_INT).SetInt(1);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom);
}
