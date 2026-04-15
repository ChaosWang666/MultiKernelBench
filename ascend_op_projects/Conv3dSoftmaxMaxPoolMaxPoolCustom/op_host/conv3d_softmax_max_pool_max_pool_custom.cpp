
#include "conv3d_softmax_max_pool_max_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = input_shape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t depth = static_cast<uint32_t>(shape[2]);
    uint32_t height = static_cast<uint32_t>(shape[3]);
    uint32_t width = static_cast<uint32_t>(shape[4]);

    uint32_t kernelSize = static_cast<uint32_t>(context->GetAttrInt("kernel_size"));
    uint32_t poolKernelSize = static_cast<uint32_t>(context->GetAttrInt("pool_kernel_size"));

    uint32_t paddedDepth = depth + 2 * (kernelSize / 2);
    uint32_t paddedHeight = height + 2 * (kernelSize / 2);
    uint32_t paddedWidth = width + 2 * (kernelSize / 2);

    uint32_t outDepth = (paddedDepth - kernelSize) / 1 + 1;
    uint32_t outHeight = (paddedHeight - kernelSize) / 1 + 1;
    uint32_t outWidth = (paddedWidth - kernelSize) / 1 + 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(static_cast<uint32_t>(context->GetAttrInt("out_channels")));
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_paddedDepth(paddedDepth);
    tiling.set_paddedHeight(paddedHeight);
    tiling.set_paddedWidth(paddedWidth);
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
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& dims = input_shape->GetOriginShape().GetDims();
    gert::Shape* output_shape = context->GetOutputShape(0);
    output_shape->SetOriginShape(gert::Shape({dims[0], static_cast<int64_t>(context->GetAttrInt("out_channels")), 
        (dims[2] + 2 * (context->GetAttrInt("kernel_size") / 2) - context->GetAttrInt("kernel_size")) / 1 + 1,
        (dims[3] + 2 * (context->GetAttrInt("kernel_size") / 2) - context->GetAttrInt("kernel_size")) / 1 + 1,
        (dims[4] + 2 * (context->GetAttrInt("kernel_size") / 2) - context->GetAttrInt("kernel_size")) / 1 + 1}));
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
class Conv3dSoftmaxMaxPoolMaxPoolCustom : public OpDef {
public:
    explicit Conv3dSoftmaxMaxPoolMaxPoolCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Attr("in_channels").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("out_channels").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("kernel_size").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("pool_kernel_size").SetType(ATTR_TYPE_INT).SetDefault(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dSoftmaxMaxPoolMaxPoolCustom);
}
