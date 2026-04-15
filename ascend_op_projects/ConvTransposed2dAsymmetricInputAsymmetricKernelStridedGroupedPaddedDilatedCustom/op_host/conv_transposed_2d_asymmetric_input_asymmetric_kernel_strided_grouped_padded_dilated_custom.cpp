
#include "conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    tiling.set_batchSize(input_dims[0]);
    tiling.set_inChannels(input_dims[1]);
    tiling.set_outChannels(weight_dims[0]);
    tiling.set_inHeight(input_dims[2]);
    tiling.set_inWidth(input_dims[3]);
    tiling.set_kernelHeight(weight_dims[2]);
    tiling.set_kernelWidth(weight_dims[3]);
    tiling.set_strideHeight(1); // Placeholder - actual values will be passed via parameters
    tiling.set_strideWidth(1); // Placeholder - actual values will be passed via parameters
    tiling.set_padHeight(0);   // Placeholder - actual values will be passed via parameters
    tiling.set_padWidth(0);    // Placeholder - actual values will be passed via parameters
    tiling.set_dilationHeight(1); // Placeholder - actual values will be passed via parameters
    tiling.set_dilationWidth(1);  // Placeholder - actual values will be passed via parameters
    tiling.set_groups(1);      // Placeholder - actual values will be passed via parameters

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
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* output_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    // Calculate output dimensions
    int64_t batch_size = input_dims[0];
    int64_t out_channels = weight_dims[0];
    int64_t in_height = input_dims[2];
    int64_t in_width = input_dims[3];
    int64_t kernel_height = weight_dims[2];
    int64_t kernel_width = weight_dims[3];

    // Placeholder values - these would normally come from parameters
    int64_t stride_h = 1;
    int64_t stride_w = 1;
    int64_t pad_h = 0;
    int64_t pad_w = 0;
    int64_t dilation_h = 1;
    int64_t dilation_w = 1;

    int64_t out_height = (in_height - 1) * stride_h - 2 * pad_h + (dilation_h * (kernel_height - 1) + 1);
    int64_t out_width = (in_width - 1) * stride_w - 2 * pad_w + (dilation_w * (kernel_width - 1) + 1);

    output_shape->SetOriginShape(gert::Shape({batch_size, out_channels, out_height, out_width}));
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
class ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom);
}
