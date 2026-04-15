
#include "conv_transposed_2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 1024;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dAsymmetricInputSquareKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* output_shape = context->GetOutputShape(0);

    uint32_t batchSize = input_shape->GetOriginShape().GetDim(0);
    uint32_t inChannels = input_shape->GetOriginShape().GetDim(1);
    uint32_t outChannels = output_shape->GetOriginShape().GetDim(1);
    uint32_t kernelH = weight_shape->GetOriginShape().GetDim(2);
    uint32_t kernelW = weight_shape->GetOriginShape().GetDim(3);
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padH = 0;
    uint32_t padW = 0;
    uint32_t outH = output_shape->GetOriginShape().GetDim(2);
    uint32_t outW = output_shape->GetOriginShape().GetDim(3);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_outH(outH);
    tiling.set_outW(outW);
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
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* output_shape = context->GetOutputShape(0);
    uint32_t batch_size = input_shape->GetOriginShape().GetDim(0);
    uint32_t in_channels = input_shape->GetOriginShape().GetDim(1);
    uint32_t kernel_h = weight_shape->GetOriginShape().GetDim(2);
    uint32_t kernel_w = weight_shape->GetOriginShape().GetDim(3);
    uint32_t out_h = input_shape->GetOriginShape().GetDim(2) * 1 + kernel_h - 1;
    uint32_t out_w = input_shape->GetOriginShape().GetDim(3) * 1 + kernel_w - 1;
    output_shape->SetOriginShape({batch_size, in_channels, out_h, out_w});
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
class ConvTransposed2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTransposed2dAsymmetricInputSquareKernelCustom);
}
