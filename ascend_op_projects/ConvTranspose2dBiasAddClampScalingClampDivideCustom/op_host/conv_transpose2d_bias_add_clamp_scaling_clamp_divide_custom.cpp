
#include "conv_transpose2d_bias_add_clamp_scaling_clamp_divide_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dBiasAddClampScalingClampDivideCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    uint32_t batchSize = input_dims[0];
    uint32_t inChannels = input_dims[1];
    uint32_t outChannels = weight_dims[1];
    uint32_t height = input_dims[2];
    uint32_t width = input_dims[3];
    uint32_t kernelH = weight_dims[2];
    uint32_t kernelW = weight_dims[3];
    uint32_t strideH = 2;
    uint32_t strideW = 2;
    uint32_t padH = 1;
    uint32_t padW = 1;
    uint32_t outputPadH = 1;
    uint32_t outputPadW = 1;
    float scalingFactor = 2.0f;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_outputPadH(outputPadH);
    tiling.set_outputPadW(outputPadW);
    tiling.set_scalingFactor(scalingFactor);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = x_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();
    gert::Shape* y_shape = context->GetOutputShape(0);
    int64_t batch_size = input_dims[0];
    int64_t out_channels = weight_dims[1];
    int64_t out_h = (input_dims[2] - 1) * 2 - 2 * 1 + 3 + 1;
    int64_t out_w = (input_dims[3] - 1) * 2 - 2 * 1 + 3 + 1;
    *y_shape = gert::Shape({batch_size, out_channels, out_h, out_w});
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
class ConvTranspose2dBiasAddClampScalingClampDivideCustom : public OpDef {
public:
    explicit ConvTranspose2dBiasAddClampScalingClampDivideCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
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

OP_ADD(ConvTranspose2dBiasAddClampScalingClampDivideCustom);
}
