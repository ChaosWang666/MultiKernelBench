
#include "conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const gert::Shape* output_shape = context->GetOutputShape(0);

    uint32_t batch = input_shape->GetOriginShape().GetDim(0);
    uint32_t inChannels = input_shape->GetOriginShape().GetDim(1);
    uint32_t outChannels = output_shape->GetOriginShape().GetDim(1);
    uint32_t depth = input_shape->GetOriginShape().GetDim(2);
    uint32_t height = input_shape->GetOriginShape().GetDim(3);
    uint32_t width = input_shape->GetOriginShape().GetDim(4);
    uint32_t kernelDepth = weight_shape->GetOriginShape().GetDim(2);
    uint32_t kernelHeight = weight_shape->GetOriginShape().GetDim(3);
    uint32_t kernelWidth = weight_shape->GetOriginShape().GetDim(4);
    uint32_t padD = 0;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t strideD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationD = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t totalLength = output_shape->GetOriginShape().GetShapeSize();

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
    tiling.set_totalLength(totalLength);
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
    gert::Shape* output_shape = context->GetOutputShape(0);
    *output_shape = *input_shape;
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
class Conv3dReluLeakyReluGeluSigmoidBiasAddCustom : public OpDef {
public:
    explicit Conv3dReluLeakyReluGeluSigmoidBiasAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIJKL})
            .UnknownShapeFormat({ge::FORMAT_OIJKL});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dReluLeakyReluGeluSigmoidBiasAddCustom);
}
