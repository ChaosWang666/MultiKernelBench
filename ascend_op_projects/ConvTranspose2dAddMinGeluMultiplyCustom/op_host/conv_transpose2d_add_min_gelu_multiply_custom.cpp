
#include "conv_transpose2d_add_min_gelu_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dAddMinGeluMultiplyCustomTilingData tiling;
    uint32_t batchSize = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t inChannels = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t height = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    uint32_t width = context->GetInputShape(0)->GetOriginShape().GetDim(3);
    uint32_t outChannels = context->GetAttrInt("out_channels");
    uint32_t kernelSize = context->GetAttrInt("kernel_size");
    uint32_t stride = context->GetAttrInt("stride");
    float addValue = context->GetAttrFloat("add_value");
    float multiplyValue = context->GetAttrFloat("multiply_value");

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_addValue(addValue);
    tiling.set_multiplyValue(multiplyValue);
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
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class ConvTranspose2dAddMinGeluMultiplyCustom : public OpDef {
public:
    explicit ConvTranspose2dAddMinGeluMultiplyCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Attr("in_channels").SetType(ATTR_VALUE_INT).SetDefault(64);
        this->Attr("out_channels").SetType(ATTR_VALUE_INT).SetDefault(128);
        this->Attr("kernel_size").SetType(ATTR_VALUE_INT).SetDefault(4);
        this->Attr("stride").SetType(ATTR_VALUE_INT).SetDefault(2);
        this->Attr("add_value").SetType(ATTR_VALUE_FLOAT).SetDefault(0.5f);
        this->Attr("multiply_value").SetType(ATTR_VALUE_FLOAT).SetDefault(2.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ConvTranspose2dAddMinGeluMultiplyCustom);
}
