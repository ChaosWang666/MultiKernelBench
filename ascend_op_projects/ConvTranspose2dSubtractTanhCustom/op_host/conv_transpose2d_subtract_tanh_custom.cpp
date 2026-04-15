
#include "conv_transpose2d_subtract_tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dSubtractTanhCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    uint32_t batchSize = static_cast<uint32_t>(input_dims[0]);
    uint32_t inChannels = static_cast<uint32_t>(input_dims[1]);
    uint32_t outChannels = static_cast<uint32_t>(weight_dims[0]);
    uint32_t height = static_cast<uint32_t>(input_dims[2]);
    uint32_t width = static_cast<uint32_t>(input_dims[3]);
    uint32_t kernelH = static_cast<uint32_t>(weight_dims[2]);
    uint32_t kernelW = static_cast<uint32_t>(weight_dims[3]);
    uint32_t strideH = 2;
    uint32_t strideW = 2;
    uint32_t padH = 1;
    uint32_t padW = 1;
    uint32_t outputPadH = 1;
    uint32_t outputPadW = 1;
    uint32_t outHeight = (height - 1) * strideH - 2 * padH + kernelH + outputPadH;
    uint32_t outWidth = (width - 1) * strideW - 2 * padW + kernelW + outputPadW;

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
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
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
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    uint32_t batchSize = static_cast<uint32_t>(input_dims[0]);
    uint32_t height = static_cast<uint32_t>(input_dims[2]);
    uint32_t width = static_cast<uint32_t>(input_dims[3]);
    uint32_t kernelH = static_cast<uint32_t>(weight_dims[2]);
    uint32_t kernelW = static_cast<uint32_t>(weight_dims[3]);
    uint32_t strideH = 2;
    uint32_t strideW = 2;
    uint32_t padH = 1;
    uint32_t padW = 1;
    uint32_t outputPadH = 1;
    uint32_t outputPadW = 1;
    uint32_t outHeight = (height - 1) * strideH - 2 * padH + kernelH + outputPadH;
    uint32_t outWidth = (width - 1) * strideW - 2 * padW + kernelW + outputPadW;

    gert::Shape* output_shape = context->GetOutputShape(0);
    output_shape->SetOriginShape({batchSize, weight_dims[0], outHeight, outWidth});
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
class ConvTranspose2dSubtractTanhCustom : public OpDef {
public:
    explicit ConvTranspose2dSubtractTanhCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose2dSubtractTanhCustom);
}
