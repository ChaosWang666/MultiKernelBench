
#include "conv_transposed_3d_square_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dSquareInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    tiling.set_batch(input_dims[0]);
    tiling.set_inChannels(input_dims[1]);
    tiling.set_outChannels(weight_dims[0]);
    tiling.set_kernelDepth(weight_dims[2]);
    tiling.set_kernelWidth(weight_dims[3]);
    tiling.set_kernelHeight(weight_dims[4]);
    tiling.set_inputDepth(input_dims[2]);
    tiling.set_inputWidth(input_dims[3]);
    tiling.set_inputHeight(input_dims[4]);
    tiling.set_strideDepth(1);
    tiling.set_strideWidth(1);
    tiling.set_strideHeight(1);
    tiling.set_padDepth(0);
    tiling.set_padWidth(0);
    tiling.set_padHeight(0);
    tiling.set_outputDepth(input_dims[2] * 1 + 0 - (weight_dims[2] - 1) * 1 - 1);
    tiling.set_outputWidth(input_dims[3] * 1 + 0 - (weight_dims[3] - 1) * 1 - 1);
    tiling.set_outputHeight(input_dims[4] * 1 + 0 - (weight_dims[4] - 1) * 1 - 1);
    tiling.set_groups(1);

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
    const std::vector<int64_t>& input_dims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = w_shape->GetOriginShape().GetDims();
    std::vector<int64_t> output_dims = {input_dims[0], weight_dims[0], input_dims[2] * 1 + 0 - (weight_dims[2] - 1) * 1 - 1, input_dims[3] * 1 + 0 - (weight_dims[3] - 1) * 1 - 1, input_dims[4] * 1 + 0 - (weight_dims[4] - 1) * 1 - 1};
    *y_shape = gert::Shape(output_dims);
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
class ConvTransposed3dSquareInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed3dSquareInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed3dSquareInputAsymmetricKernelCustom);
}
