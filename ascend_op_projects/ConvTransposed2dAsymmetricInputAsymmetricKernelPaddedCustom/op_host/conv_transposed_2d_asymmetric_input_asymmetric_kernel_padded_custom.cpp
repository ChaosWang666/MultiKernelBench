
#include "conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputAsymmetricKernelPaddedCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape& outputShape = context->GetOutputShape(0);
    
    tiling.set_batchSize(outputShape.GetDim(0));
    tiling.set_inChannels(inputShape->GetDim(1));
    tiling.set_outChannels(weightShape->GetDim(0));
    tiling.set_inputHeight(inputShape->GetDim(2));
    tiling.set_inputWidth(inputShape->GetDim(3));
    tiling.set_kernelHeight(weightShape->GetDim(2));
    tiling.set_kernelWidth(weightShape->GetDim(3));
    tiling.set_strideHeight(1); // Assuming stride is fixed for simplicity
    tiling.set_strideWidth(1);
    tiling.set_padHeight(1); // Assuming padding is fixed for simplicity
    tiling.set_padWidth(1);
    tiling.set_outputHeight(outputShape.GetDim(2));
    tiling.set_outputWidth(outputShape.GetDim(3));

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
    
    uint32_t batch = inputShape->GetDim(0);
    uint32_t outChannels = weightShape->GetDim(0);
    uint32_t inputH = inputShape->GetDim(2);
    uint32_t inputW = inputShape->GetDim(3);
    uint32_t kernelH = weightShape->GetDim(2);
    uint32_t kernelW = weightShape->GetDim(3);
    uint32_t padH = 1;
    uint32_t padW = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    
    uint32_t outH = (inputH - 1) * strideH - 2 * padH + kernelH;
    uint32_t outW = (inputW - 1) * strideW - 2 * padW + kernelW;
    
    outputShape->SetDim(0, batch);
    outputShape->SetDim(1, outChannels);
    outputShape->SetDim(2, outH);
    outputShape->SetDim(3, outW);
    
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
class ConvTransposed2dAsymmetricInputAsymmetricKernelPaddedCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelPaddedCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("weight")
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

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelPaddedCustom);
}
