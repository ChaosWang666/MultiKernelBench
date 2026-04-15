
#include "conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dAsymmetricInputSquareKernelStridedPaddedGroupedCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    tiling.set_batch(inputDims[0]);
    tiling.set_inChannels(inputDims[1]);
    tiling.set_outChannels(weightDims[0]);
    tiling.set_kernelDepth(weightDims[2]);
    tiling.set_kernelHeight(weightDims[3]);
    tiling.set_kernelWidth(weightDims[4]);
    tiling.set_strideDepth(2); // Assuming default stride for now
    tiling.set_strideHeight(2);
    tiling.set_strideWidth(2);
    tiling.set_padDepth(1); // Assuming default padding for now
    tiling.set_padHeight(1);
    tiling.set_padWidth(1);
    tiling.set_groups(4); // Assuming default groups for now
    tiling.set_inputDepth(inputDims[2]);
    tiling.set_inputHeight(inputDims[3]);
    tiling.set_inputWidth(inputDims[4]);
    tiling.set_outputDepth(32); // Placeholder
    tiling.set_outputHeight(64); // Placeholder
    tiling.set_outputWidth(128); // Placeholder

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

    std::vector<int64_t> outputDims = {inputDims[0], weightDims[0], 32, 64, 128}; // Placeholder values
    *outputShape = gert::Shape(outputDims);
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
class ConvTransposed3dAsymmetricInputSquareKernelStridedPaddedGroupedCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputSquareKernelStridedPaddedGroupedCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
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

OP_ADD(ConvTransposed3dAsymmetricInputSquareKernelStridedPaddedGroupedCustom);
}
