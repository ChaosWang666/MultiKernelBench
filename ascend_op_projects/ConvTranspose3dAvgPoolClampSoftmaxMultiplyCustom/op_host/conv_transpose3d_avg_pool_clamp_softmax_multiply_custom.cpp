
#include "conv_transpose3d_avg_pool_clamp_softmax_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    uint32_t batchSize = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t depth = static_cast<uint32_t>(shape[2]);
    uint32_t height = static_cast<uint32_t>(shape[3]);
    uint32_t width = static_cast<uint32_t>(shape[4]);

    uint32_t outChannels = 64;
    uint32_t kernelSize = 3;
    uint32_t stride = 2;
    uint32_t padding = 1;
    uint32_t outputPadding = 1;
    uint32_t poolKernelSize = 2;
    float clampMin = 0.0f;
    float clampMax = 1.0f;

    uint32_t totalElements = batchSize * inChannels * depth * height * width;
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_outputPadding(outputPadding);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_clampMin(clampMin);
    tiling.set_clampMax(clampMax);
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
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom : public OpDef {
public:
    explicit ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(ConvTranspose3dAvgPoolClampSoftmaxMultiplyCustom);
}
