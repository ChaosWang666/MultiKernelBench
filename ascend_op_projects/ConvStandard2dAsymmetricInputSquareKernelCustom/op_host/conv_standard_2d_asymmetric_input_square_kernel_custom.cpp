
#include "conv_standard_2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dAsymmetricInputSquareKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    uint32_t batch = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t height = static_cast<uint32_t>(inputDims[2]);
    uint32_t width = static_cast<uint32_t>(inputDims[3]);
    uint32_t outChannels = static_cast<uint32_t>(weightDims[0]);
    uint32_t kernelH = static_cast<uint32_t>(weightDims[2]);
    uint32_t kernelW = static_cast<uint32_t>(weightDims[3]);
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t groups = 1;
    uint32_t outHeight = (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    uint32_t outWidth = (width + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
    tiling.set_groups(groups);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& inputDims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = w_shape->GetOriginShape().GetDims();

    uint32_t batch = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t height = static_cast<uint32_t>(inputDims[2]);
    uint32_t width = static_cast<uint32_t>(inputDims[3]);
    uint32_t outChannels = static_cast<uint32_t>(weightDims[0]);
    uint32_t kernelH = static_cast<uint32_t>(weightDims[2]);
    uint32_t kernelW = static_cast<uint32_t>(weightDims[3]);
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t outHeight = (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    uint32_t outWidth = (width + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;

    y_shape->SetOriginShape(gert::Shape({batch, outChannels, outHeight, outWidth}));
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
class ConvStandard2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvStandard2dAsymmetricInputSquareKernelCustom);
}
