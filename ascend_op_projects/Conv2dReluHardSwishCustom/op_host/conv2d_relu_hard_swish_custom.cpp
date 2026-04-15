
#include "conv2d_relu_hard_swish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dReluHardSwishCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const gert::Shape* output_shape = context->GetOutputShape(0);

    tiling.set_batchSize(input_shape->GetDim(0));
    tiling.set_inputChannels(input_shape->GetDim(1));
    tiling.set_outputChannels(weight_shape->GetDim(0));
    tiling.set_inputHeight(input_shape->GetDim(2));
    tiling.set_inputWidth(input_shape->GetDim(3));
    tiling.set_kernelHeight(weight_shape->GetDim(2));
    tiling.set_kernelWidth(weight_shape->GetDim(3));
    tiling.set_padH(0);
    tiling.set_padW(0);
    tiling.set_strideH(1);
    tiling.set_strideW(1);
    tiling.set_dilationH(1);
    tiling.set_dilationW(1);

    context->SetBlockDim(BLOCK_DIM);
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
    output_shape->SetDim(0, input_shape->GetDim(0));
    output_shape->SetDim(1, weight_shape->GetDim(0));
    output_shape->SetDim(2, input_shape->GetDim(2));
    output_shape->SetDim(3, input_shape->GetDim(3));
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
class Conv2dReluHardSwishCustom : public OpDef {
public:
    explicit Conv2dReluHardSwishCustom(const char* name) : OpDef(name)
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
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dReluHardSwishCustom);
}
