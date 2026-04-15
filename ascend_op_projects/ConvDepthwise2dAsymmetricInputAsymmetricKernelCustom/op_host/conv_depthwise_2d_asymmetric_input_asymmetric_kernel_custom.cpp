
#include "conv_depthwise_2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM_H = 16;
const uint32_t TILE_NUM_W = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    uint32_t batchSize = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t inputHeight = static_cast<uint32_t>(inputDims[2]);
    uint32_t inputWidth = static_cast<uint32_t>(inputDims[3]);
    uint32_t outChannels = static_cast<uint32_t>(weightDims[0]);
    uint32_t kernelHeight = static_cast<uint32_t>(weightDims[2]);
    uint32_t kernelWidth = static_cast<uint32_t>(weightDims[3]);
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t outputHeight = (inputHeight + 2 * padH - (dilationH * (kernelHeight - 1) + 1)) / strideH + 1;
    uint32_t outputWidth = (inputWidth + 2 * padW - (dilationW * (kernelWidth - 1) + 1)) / strideW + 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_tileNumH(TILE_NUM_H);
    tiling.set_tileNumW(TILE_NUM_W);
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
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    uint32_t batchSize = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t inputHeight = static_cast<uint32_t>(inputDims[2]);
    uint32_t inputWidth = static_cast<uint32_t>(inputDims[3]);
    uint32_t outChannels = static_cast<uint32_t>(weightDims[0]);
    uint32_t kernelHeight = static_cast<uint32_t>(weightDims[2]);
    uint32_t kernelWidth = static_cast<uint32_t>(weightDims[3]);
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t outputHeight = (inputHeight + 2 * padH - (dilationH * (kernelHeight - 1) + 1)) / strideH + 1;
    uint32_t outputWidth = (inputWidth + 2 * padW - (dilationW * (kernelWidth - 1) + 1)) / strideW + 1;

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->GetOriginShape().SetDims({static_cast<int64_t>(batchSize), static_cast<int64_t>(outChannels), static_cast<int64_t>(outputHeight), static_cast<int64_t>(outputWidth)});
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
class ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom);
}
