
#include "conv2d_subtract_hard_swish_max_pool_mish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dSubtractHardSwishMaxPoolMishCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    uint32_t batchSize = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t height = static_cast<uint32_t>(shape[2]);
    uint32_t width = static_cast<uint32_t>(shape[3]);

    uint32_t outChannels = context->GetAttrInt("out_channels");
    uint32_t kernelSize = context->GetAttrInt("kernel_size");
    float subtractValue = context->GetAttrFloat("subtract_value");
    uint32_t poolKernelSize = context->GetAttrInt("pool_kernel_size");

    uint32_t totalElements = batchSize * outChannels * (height - kernelSize + 1) * (width - kernelSize + 1);
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_subtractValue(subtractValue);
    tiling.set_poolKernelSize(poolKernelSize);
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
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& dims = inputShape->GetOriginShape().GetDims();
    uint32_t batchSize = static_cast<uint32_t>(dims[0]);
    uint32_t inChannels = static_cast<uint32_t>(dims[1]);
    uint32_t height = static_cast<uint32_t>(dims[2]);
    uint32_t width = static_cast<uint32_t>(dims[3]);
    uint32_t outChannels = context->GetAttrInt("out_channels");
    uint32_t kernelSize = context->GetAttrInt("kernel_size");
    uint32_t poolKernelSize = context->GetAttrInt("pool_kernel_size");

    std::vector<int64_t> outputDims = {static_cast<int64_t>(batchSize), static_cast<int64_t>(outChannels),
                                       static_cast<int64_t>(height - kernelSize + 1),
                                       static_cast<int64_t>(width - kernelSize + 1)};
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = gert::Shape(outputDims);
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
class Conv2dSubtractHardSwishMaxPoolMishCustom : public OpDef {
public:
    explicit Conv2dSubtractHardSwishMaxPoolMishCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Attr("in_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("kernel_size").SetType(INT).SetParamType(REQUIRED);
        this->Attr("subtract_value").SetType(FLOAT).SetParamType(REQUIRED);
        this->Attr("pool_kernel_size").SetType(INT).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dSubtractHardSwishMaxPoolMishCustom);
}
