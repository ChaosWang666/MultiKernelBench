
#include "conv_transposed_2d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    tiling.set_batchSize(input_dims[0]);
    tiling.set_inChannels(input_dims[1]);
    tiling.set_outChannels(weight_dims[0]);
    tiling.set_heightIn(input_dims[2]);
    tiling.set_widthIn(input_dims[3]);
    tiling.set_kernelHeight(weight_dims[2]);
    tiling.set_kernelWidth(weight_dims[3]);
    tiling.set_strideH(1); // Assuming default stride
    tiling.set_strideW(1); // Assuming default stride
    tiling.set_padH(0); // Assuming default padding
    tiling.set_padW(0); // Assuming default padding
    tiling.set_dilationH(1); // Assuming default dilation
    tiling.set_dilationW(1); // Assuming default dilation
    tiling.set_groups(1); // Assuming default groups

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
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& input_dims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    std::vector<int64_t> output_dims = {input_dims[0], weight_dims[0], 1, 1}; // Placeholder values
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
class ConvTransposed2dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelCustom);
}
