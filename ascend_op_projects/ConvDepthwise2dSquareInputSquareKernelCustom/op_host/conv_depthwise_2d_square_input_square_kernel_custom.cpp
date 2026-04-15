
#include "conv_depthwise_2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvDepthwise2dSquareInputSquareKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape& outputShape = context->GetOutputShape(0);
    
    uint32_t batch = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t outHeight = outputShape.GetDim(2);
    uint32_t outWidth = outputShape.GetDim(3);
    uint32_t kernelH = weightShape->GetOriginShape().GetDim(2);
    uint32_t kernelW = weightShape->GetOriginShape().GetDim(3);
    uint32_t strideH = context->GetAttrInt("stride_h");
    uint32_t strideW = context->GetAttrInt("stride_w");
    uint32_t padH = context->GetAttrInt("pad_h");
    uint32_t padW = context->GetAttrInt("pad_w");
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
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
    gert::Shape* outputShape = context->GetOutputShape(0);
    
    uint32_t batch = inputShape->GetOriginShape().GetDim(0);
    uint32_t inChannels = inputShape->GetOriginShape().GetDim(1);
    uint32_t inHeight = inputShape->GetOriginShape().GetDim(2);
    uint32_t inWidth = inputShape->GetOriginShape().GetDim(3);
    uint32_t kernelH = weightShape->GetOriginShape().GetDim(2);
    uint32_t kernelW = weightShape->GetOriginShape().GetDim(3);
    uint32_t strideH = context->GetAttrInt("stride_h");
    uint32_t strideW = context->GetAttrInt("stride_w");
    uint32_t padH = context->GetAttrInt("pad_h");
    uint32_t padW = context->GetAttrInt("pad_w");
    
    uint32_t outHeight = (inHeight + 2 * padH - kernelH) / strideH + 1;
    uint32_t outWidth = (inWidth + 2 * padW - kernelW) / strideW + 1;
    
    outputShape->SetDim(0, batch);
    outputShape->SetDim(1, inChannels);
    outputShape->SetDim(2, outHeight);
    outputShape->SetDim(3, outWidth);
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
class ConvDepthwise2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
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
            
        this->Attr("stride_h").SetType(INT).SetDefault(1);
        this->Attr("stride_w").SetType(INT).SetDefault(1);
        this->Attr("pad_h").SetType(INT).SetDefault(0);
        this->Attr("pad_w").SetType(INT).SetDefault(0);
        
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvDepthwise2dSquareInputSquareKernelCustom);
}
