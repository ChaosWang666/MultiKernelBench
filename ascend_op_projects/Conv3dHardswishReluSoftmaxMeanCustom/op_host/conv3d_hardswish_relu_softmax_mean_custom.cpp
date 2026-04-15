
#include "conv3d_hardswish_relu_softmax_mean_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dHardswishReluSoftmaxMeanCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    const gert::Shape* outputShape = context->GetOutputShape(0);

    uint32_t batch = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t outChannels = outputShape->GetOriginShape().GetDim(1);
    uint32_t depth = inputShape->GetOriginShape().GetDim(2);
    uint32_t height = inputShape->GetOriginShape().GetDim(3);
    uint32_t width = inputShape->GetOriginShape().GetDim(4);
    uint32_t kernelDepth = weightShape->GetOriginShape().GetDim(2);
    uint32_t kernelHeight = weightShape->GetOriginShape().GetDim(3);
    uint32_t kernelWidth = weightShape->GetOriginShape().GetDim(4);
    uint32_t padD = 0;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t strideD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationD = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t group = 1;
    uint32_t totalElements = batch * outChannels;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelDepth(kernelDepth);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_padD(padD);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_strideD(strideD);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_dilationD(dilationD);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
    tiling.set_group(group);
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
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDim(0, inputShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(1, weightShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(2, 1); // After mean pooling
    outputShape->SetDim(3, 1);
    outputShape->SetDim(4, 1);
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
class Conv3dHardswishReluSoftmaxMeanCustom : public OpDef {
public:
    explicit Conv3dHardswishReluSoftmaxMeanCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
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

OP_ADD(Conv3dHardswishReluSoftmaxMeanCustom);
}
