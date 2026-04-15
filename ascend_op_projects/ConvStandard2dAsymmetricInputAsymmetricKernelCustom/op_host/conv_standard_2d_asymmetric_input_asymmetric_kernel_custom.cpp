
#include "conv_standard_2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* outputShape = context->GetOutputShape(0);

    tiling.set_batchSize(outputShape->GetOriginShape().GetDim(0));
    tiling.set_inputChannels(inputShape->GetOriginShape().GetDim(1));
    tiling.set_outputChannels(weightShape->GetOriginShape().GetDim(0));
    tiling.set_inputHeight(inputShape->GetOriginShape().GetDim(2));
    tiling.set_inputWidth(inputShape->GetOriginShape().GetDim(3));
    tiling.set_kernelHeight(weightShape->GetOriginShape().GetDim(2));
    tiling.set_kernelWidth(weightShape->GetOriginShape().GetDim(3));
    tiling.set_strideH(1); // Assuming default stride for now
    tiling.set_strideW(1);
    tiling.set_padH(0); // Assuming default padding for now
    tiling.set_padW(0);
    tiling.set_dilationH(1); // Assuming default dilation for now
    tiling.set_dilationW(1);
    tiling.set_outputHeight(outputShape->GetOriginShape().GetDim(2));
    tiling.set_outputWidth(outputShape->GetOriginShape().GetDim(3));
    tiling.set_groups(1); // Assuming default groups for now

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
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDim(0, inputShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(1, weightShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(2, (inputShape->GetOriginShape().GetDim(2) + 2 * 0 - (1 + (weightShape->GetOriginShape().GetDim(2) - 1) * 1)) / 1 + 1);
    outputShape->SetDim(3, (inputShape->GetOriginShape().GetDim(3) + 2 * 0 - (1 + (weightShape->GetOriginShape().GetDim(3) - 1) * 1)) / 1 + 1);
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
class ConvStandard2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvStandard2dAsymmetricInputAsymmetricKernelCustom);
}
