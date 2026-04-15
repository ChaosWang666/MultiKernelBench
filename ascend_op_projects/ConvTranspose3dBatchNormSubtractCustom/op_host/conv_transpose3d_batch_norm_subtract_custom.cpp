
#include "conv_transpose3d_batch_norm_subtract_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dBatchNormSubtractCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    uint32_t batchSize = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t depth = static_cast<uint32_t>(shape[2]);
    uint32_t height = static_cast<uint32_t>(shape[3]);
    uint32_t width = static_cast<uint32_t>(shape[4]);

    uint32_t outDepth = (depth - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");
    uint32_t outHeight = (height - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");
    uint32_t outWidth = (width - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");

    uint32_t totalElements = batchSize * context->GetAttrInt("out_channels") * outDepth * outHeight * outWidth;
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));
    tiling.set_totalElements(totalElements);
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
    const std::vector<int64_t>& inputDims = x1_shape->GetOriginShape().GetDims();
    gert::Shape* y_shape = context->GetOutputShape(0);
    int64_t outDepth = (inputDims[2] - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");
    int64_t outHeight = (inputDims[3] - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");
    int64_t outWidth = (inputDims[4] - 1) * context->GetAttrInt("stride") + context->GetAttrInt("kernel_size") - 2 * context->GetAttrInt("padding");
    y_shape->GetOriginShape().SetDims({inputDims[0], context->GetAttrInt("out_channels"), outDepth, outHeight, outWidth});
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
class ConvTranspose3dBatchNormSubtractCustom : public OpDef {
public:
    explicit ConvTranspose3dBatchNormSubtractCustom(const char* name) : OpDef(name)
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
        this->Attr("in_channels").SetType(INT).SetDefault(0);
        this->Attr("out_channels").SetType(INT).SetDefault(0);
        this->Attr("kernel_size").SetType(INT).SetDefault(0);
        this->Attr("stride").SetType(INT).SetDefault(0);
        this->Attr("padding").SetType(INT).SetDefault(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dBatchNormSubtractCustom);
}
