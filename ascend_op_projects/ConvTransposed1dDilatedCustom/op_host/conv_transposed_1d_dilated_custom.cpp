
#include "conv_transposed_1d_dilated_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ConvTransposed1dDilatedCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* y_shape = context->GetOutputShape(0);
    
    uint32_t batchSize = x_shape->GetOriginShape().GetDim(0);
    uint32_t inChannels = x_shape->GetOriginShape().GetDim(1);
    uint32_t outChannels = weight_shape->GetOriginShape().GetDim(0);
    uint32_t inputLength = x_shape->GetOriginShape().GetDim(2);
    uint32_t outputLength = y_shape->GetOriginShape().GetDim(2);
    uint32_t kernelSize = weight_shape->GetOriginShape().GetDim(1);
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_inputLength(inputLength);
    tiling.set_outputLength(outputLength);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_dilation(dilation);
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
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    uint32_t batchSize = x1_shape->GetOriginShape().GetDim(0);
    uint32_t outChannels = weight_shape->GetOriginShape().GetDim(0);
    uint32_t inputLength = x1_shape->GetOriginShape().GetDim(2);
    uint32_t kernelSize = weight_shape->GetOriginShape().GetDim(1);
    uint32_t stride = 1;
    uint32_t padding = 0;
    uint32_t dilation = 1;
    uint32_t outputLength = (inputLength - 1) * stride - 2 * padding + (dilation * (kernelSize - 1) + 1);
    y_shape->SetDim(0, batchSize);
    y_shape->SetDim(1, outChannels);
    y_shape->SetDim(2, outputLength);
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
class ConvTransposed1dDilatedCustom : public OpDef {
public:
    explicit ConvTransposed1dDilatedCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed1dDilatedCustom);
}
