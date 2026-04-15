
#include "conv_standard_2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 1024;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard2dSquareInputSquareKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* output_shape = context->GetOutputShape(0);

    uint32_t batchSize = input_shape->GetOriginShape().GetDim(0);
    uint32_t inputHeight = input_shape->GetOriginShape().GetDim(2);
    uint32_t inputWidth = input_shape->GetOriginShape().GetDim(3);
    uint32_t inputChannel = input_shape->GetOriginShape().GetDim(1);
    uint32_t outputChannel = output_shape->GetOriginShape().GetDim(1);
    uint32_t kernelHeight = weight_shape->GetOriginShape().GetDim(2);
    uint32_t kernelWidth = weight_shape->GetOriginShape().GetDim(3);
    uint32_t strideH = 4;
    uint32_t strideW = 4;
    uint32_t padH = 2;
    uint32_t padW = 2;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t outputHeight = (inputHeight + 2 * padH - (dilationH * (kernelHeight - 1) + 1)) / strideH + 1;
    uint32_t outputWidth = (inputWidth + 2 * padW - (dilationW * (kernelWidth - 1) + 1)) / strideW + 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_inputChannel(inputChannel);
    tiling.set_outputChannel(outputChannel);
    tiling.set_kernelHeight(kernelHeight);
    tiling.set_kernelWidth(kernelWidth);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_tileNum(TILE_NUM);
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
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    uint32_t batch = x1_shape->GetOriginShape().GetDim(0);
    uint32_t out_channel = w_shape->GetOriginShape().GetDim(0);
    uint32_t h = x1_shape->GetOriginShape().GetDim(2);
    uint32_t w = x1_shape->GetOriginShape().GetDim(3);
    y_shape->SetOriginShape(gert::Shape::CreateShape({batch, out_channel, h, w}));
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
class ConvStandard2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard2dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("w")
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

OP_ADD(ConvStandard2dSquareInputSquareKernelCustom);
}
