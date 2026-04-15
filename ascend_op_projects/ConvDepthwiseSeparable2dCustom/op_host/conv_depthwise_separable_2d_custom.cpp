
#include "conv_depthwise_separable_2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwiseSeparable2dCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* depthwiseWeightShape = context->GetInputShape(1);
    const gert::Shape* pointwiseWeightShape = context->GetInputShape(2);
    
    uint32_t batchSize = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t height = inputShape->GetOriginShape().GetDim(2);
    uint32_t width = inputShape->GetOriginShape().GetDim(3);
    uint32_t outChannels = pointwiseWeightShape->GetOriginShape().GetDim(0);
    uint32_t kernelSize = depthwiseWeightShape->GetOriginShape().GetDim(2);
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
    
    uint32_t depthwiseOutHeight = (height + 2 * padding - (dilation * (kernelSize - 1) + 1)) / stride + 1;
    uint32_t depthwiseOutWidth = (width + 2 * padding - (dilation * (kernelSize - 1) + 1)) / stride + 1;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);
    tiling.set_depthwiseOutHeight(depthwiseOutHeight);
    tiling.set_depthwiseOutWidth(depthwiseOutWidth);
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
    const gert::Shape* depthwiseWeightShape = context->GetInputShape(1);
    const gert::Shape* pointwiseWeightShape = context->GetInputShape(2);
    
    uint32_t batchSize = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t height = inputShape->GetOriginShape().GetDim(2);
    uint32_t width = inputShape->GetOriginShape().GetDim(3);
    uint32_t outChannels = pointwiseWeightShape->GetOriginShape().GetDim(0);
    uint32_t kernelSize = depthwiseWeightShape->GetOriginShape().GetDim(2);
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
    
    uint32_t depthwiseOutHeight = (height + 2 * padding - (dilation * (kernelSize - 1) + 1)) / stride + 1;
    uint32_t depthwiseOutWidth = (width + 2 * padding - (dilation * (kernelSize - 1) + 1)) / stride + 1;
    
    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetOriginShape({batchSize, outChannels, depthwiseOutHeight, depthwiseOutWidth});
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
class ConvDepthwiseSeparable2dCustom : public OpDef {
public:
    explicit ConvDepthwiseSeparable2dCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("depthwiseWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("pointwiseWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvDepthwiseSeparable2dCustom);
}
