
#include "conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    tiling.set_batchSize(inputDims[0]);
    tiling.set_inChannels(inputDims[1]);
    tiling.set_outChannels(weightDims[0]);
    tiling.set_heightIn(inputDims[2]);
    tiling.set_widthIn(inputDims[3]);
    tiling.set_kernelSize(weightDims[2]);
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));
    tiling.set_dilation(context->GetAttrInt("dilation"));

    // Calculate output dimensions
    int64_t heightOut = (inputDims[2] - 1) * tiling.stride() - 2 * tiling.padding() + (tiling.kernelSize() - 1) * tiling.dilation() + 1;
    int64_t widthOut = (inputDims[3] - 1) * tiling.stride() - 2 * tiling.padding() + (tiling.kernelSize() - 1) * tiling.dilation() + 1;
    tiling.set_heightOut(heightOut);
    tiling.set_widthOut(widthOut);

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
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    int64_t batchSize = inputDims[0];
    int64_t outChannels = weightDims[0];
    int64_t heightIn = inputDims[2];
    int64_t widthIn = inputDims[3];
    int64_t kernelSize = weightDims[2];
    int64_t stride = context->GetAttrInt("stride");
    int64_t padding = context->GetAttrInt("padding");
    int64_t dilation = context->GetAttrInt("dilation");
    int64_t heightOut = (heightIn - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;
    int64_t widthOut = (widthIn - 1) * stride - 2 * padding + (kernelSize - 1) * dilation + 1;

    outputShape->SetDims({batchSize, outChannels, heightOut, widthOut});
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
class ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom);
}
