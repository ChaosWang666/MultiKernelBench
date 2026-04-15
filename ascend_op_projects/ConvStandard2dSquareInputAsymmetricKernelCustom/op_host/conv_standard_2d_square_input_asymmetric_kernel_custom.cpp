
#include "conv_standard_2d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dSquareInputAsymmetricKernelCustomTilingData tiling;
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
    tiling.set_padH(0); // Assuming zero padding for simplicity
    tiling.set_padW(0);
    tiling.set_strideH(1); // Assuming stride 1 for simplicity
    tiling.set_strideW(1);
    tiling.set_dilationH(1); // Assuming dilation 1 for simplicity
    tiling.set_dilationW(1);
    tiling.set_groups(1); // Assuming groups 1 for simplicity
    tiling.set_outHeight((inputDims[2] + 2 * 0 - (weightDims[2] - 1) * 1) / 1 + 1);
    tiling.set_outWidth((inputDims[3] + 2 * 0 - (weightDims[3] - 1) * 1) / 1 + 1);

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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& inputDims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = w_shape->GetOriginShape().GetDims();
    std::vector<int64_t> outputDims = {inputDims[0], weightDims[0], (inputDims[2] + 2 * 0 - (weightDims[2] - 1) * 1) / 1 + 1, (inputDims[3] + 2 * 0 - (weightDims[3] - 1) * 1) / 1 + 1};
    *y_shape = gert::Shape(outputDims);
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
class ConvStandard2dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dSquareInputAsymmetricKernelCustom);
}
