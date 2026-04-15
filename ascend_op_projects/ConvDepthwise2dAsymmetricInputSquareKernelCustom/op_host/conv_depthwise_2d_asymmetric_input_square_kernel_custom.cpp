
#include "conv_depthwise_2d_asymmetric_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ConvDepthwise2dAsymmetricInputSquareKernelCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    const gert::Shape* y_shape = context->GetOutputShape(0);
    
    tiling.set_batchSize(x_shape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(x_shape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(y_shape->GetOriginShape().GetDim(1));
    tiling.set_heightIn(x_shape->GetOriginShape().GetDim(2));
    tiling.set_widthIn(x_shape->GetOriginShape().GetDim(3));
    tiling.set_heightOut(y_shape->GetOriginShape().GetDim(2));
    tiling.set_widthOut(y_shape->GetOriginShape().GetDim(3));
    tiling.set_kernelSize(w_shape->GetOriginShape().GetDim(2));
    tiling.set_stride(1); // Assuming stride is 1 for simplicity
    tiling.set_padding(0); // Assuming padding is 0 for simplicity
    
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    
    uint32_t batch_size = x1_shape->GetOriginShape().GetDim(0);
    uint32_t in_channels = x1_shape->GetOriginShape().GetDim(1);
    uint32_t height_in = x1_shape->GetOriginShape().GetDim(2);
    uint32_t width_in = x1_shape->GetOriginShape().GetDim(3);
    uint32_t kernel_size = w_shape->GetOriginShape().GetDim(2);
    uint32_t stride = 1; // Assuming stride is 1 for simplicity
    uint32_t padding = 0; // Assuming padding is 0 for simplicity
    
    uint32_t height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    uint32_t width_out = (width_in + 2 * padding - kernel_size) / stride + 1;
    
    y_shape->GetOriginShape().SetDim(0, batch_size);
    y_shape->GetOriginShape().SetDim(1, in_channels);
    y_shape->GetOriginShape().SetDim(2, height_out);
    y_shape->GetOriginShape().SetDim(3, width_out);
    
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
class ConvDepthwise2dAsymmetricInputSquareKernelCustom : public OpDef {
public:
    explicit ConvDepthwise2dAsymmetricInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvDepthwise2dAsymmetricInputSquareKernelCustom);
}
