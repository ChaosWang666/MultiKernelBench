
#include "conv_transposed_3d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dSquareInputSquareKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t depth = static_cast<uint32_t>(inputDims[2]);
    uint32_t height = static_cast<uint32_t>(inputDims[3]);
    uint32_t width = static_cast<uint32_t>(inputDims[4]);

    uint32_t outChannels = static_cast<uint32_t>(context->GetAttrInt("out_channels"));
    uint32_t kernelSize = static_cast<uint32_t>(context->GetAttrInt("kernel_size"));
    uint32_t stride = static_cast<uint32_t>(context->GetAttrInt("stride"));
    uint32_t padding = static_cast<uint32_t>(context->GetAttrInt("padding"));
    uint32_t outputPadding = static_cast<uint32_t>(context->GetAttrInt("output_padding"));
    uint32_t groups = static_cast<uint32_t>(context->GetAttrInt("groups"));

    uint32_t outDepth = (depth - 1) * stride - 2 * padding + kernelSize + outputPadding;
    uint32_t outHeight = (height - 1) * stride - 2 * padding + kernelSize + outputPadding;
    uint32_t outWidth = (width - 1) * stride - 2 * padding + kernelSize + outputPadding;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_outputPadding(outputPadding);
    tiling.set_groups(groups);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_outDepth(outDepth);
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
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(inputDims[0]);
    uint32_t inChannels = static_cast<uint32_t>(inputDims[1]);
    uint32_t depth = static_cast<uint32_t>(inputDims[2]);
    uint32_t height = static_cast<uint32_t>(inputDims[3]);
    uint32_t width = static_cast<uint32_t>(inputDims[4]);

    uint32_t outChannels = static_cast<uint32_t>(context->GetAttrInt("out_channels"));
    uint32_t kernelSize = static_cast<uint32_t>(context->GetAttrInt("kernel_size"));
    uint32_t stride = static_cast<uint32_t>(context->GetAttrInt("stride"));
    uint32_t padding = static_cast<uint32_t>(context->GetAttrInt("padding"));
    uint32_t outputPadding = static_cast<uint32_t>(context->GetAttrInt("output_padding"));

    uint32_t outDepth = (depth - 1) * stride - 2 * padding + kernelSize + outputPadding;
    uint32_t outHeight = (height - 1) * stride - 2 * padding + kernelSize + outputPadding;
    uint32_t outWidth = (width - 1) * stride - 2 * padding + kernelSize + outputPadding;

    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetOriginShape({batch, outChannels, outDepth, outHeight, outWidth});
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
class ConvTransposed3dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed3dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Attr("in_channels").ParamType(REQUIRED).DataType(ge::DT_INT32);
        this->Attr("out_channels").ParamType(REQUIRED).DataType(ge::DT_INT32);
        this->Attr("kernel_size").ParamType(REQUIRED).DataType(ge::DT_INT32);
        this->Attr("stride").ParamType(OPTIONAL).DataType(ge::DT_INT32).DefaultValue(1);
        this->Attr("padding").ParamType(OPTIONAL).DataType(ge::DT_INT32).DefaultValue(0);
        this->Attr("output_padding").ParamType(OPTIONAL).DataType(ge::DT_INT32).DefaultValue(0);
        this->Attr("groups").ParamType(OPTIONAL).DataType(ge::DT_INT32).DefaultValue(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed3dSquareInputSquareKernelCustom);
}
