
#include "conv_transposed_2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dSquareInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    tiling.set_batchSize(inputDims[0]);
    tiling.set_inChannels(inputDims[1]);
    tiling.set_outChannels(weightDims[0]);
    tiling.set_height(inputDims[2]);
    tiling.set_width(inputDims[3]);
    tiling.set_kernelH(weightDims[2]);
    tiling.set_kernelW(weightDims[3]);
    tiling.set_strideH(1); // Assuming default stride for simplicity
    tiling.set_strideW(1);
    tiling.set_padH(0); // Assuming default padding for simplicity
    tiling.set_padW(0);
    tiling.set_outputPadH(0); // Assuming default output padding for simplicity
    tiling.set_outputPadW(0);
    tiling.set_groups(1); // Assuming default groups for simplicity

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
    int64_t height = inputDims[2];
    int64_t width = inputDims[3];

    // Simplified output size calculation
    int64_t outHeight = (height - 1) * 1 + weightDims[2] - 0 + 0;
    int64_t outWidth = (width - 1) * 1 + weightDims[3] - 0 + 0;

    outputShape->SetOriginShape(gert::Shape({batchSize, outChannels, outHeight, outWidth}));
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
class ConvTransposed2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dSquareInputAsymmetricKernelCustom);
}
