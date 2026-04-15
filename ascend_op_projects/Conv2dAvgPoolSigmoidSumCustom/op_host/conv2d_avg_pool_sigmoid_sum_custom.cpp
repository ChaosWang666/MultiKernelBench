
#include "conv2d_avg_pool_sigmoid_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dAvgPoolSigmoidSumCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    uint32_t batchSize = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t outChannels = weightShape->GetOriginShape().GetDim(0);
    uint32_t height = inputShape->GetOriginShape().GetDim(2);
    uint32_t width = inputShape->GetOriginShape().GetDim(3);
    uint32_t kernelH = weightShape->GetOriginShape().GetDim(2);
    uint32_t kernelW = weightShape->GetOriginShape().GetDim(3);
    uint32_t poolH = 4;
    uint32_t poolW = 4;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t outputHeight = (height + 2 * padH - kernelH) / strideH + 1;
    uint32_t outputWidth = (width + 2 * padW - kernelW) / strideW + 1;
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_poolH(poolH);
    tiling.set_poolW(poolW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
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
    const gert::Shape* weightShape = context->GetInputShape(1);
    uint32_t batchSize = inputShape->GetOriginShape().GetDim(0);
    uint32_t outChannels = weightShape->GetOriginShape().GetDim(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDim(0, batchSize);
    y_shape->SetDim(1, outChannels);
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
class Conv2dAvgPoolSigmoidSumCustom : public OpDef {
public:
    explicit Conv2dAvgPoolSigmoidSumCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW})
            .UnknownShapeFormat({ge::FORMAT_OIHW});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("y")
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

OP_ADD(Conv2dAvgPoolSigmoidSumCustom);
}
