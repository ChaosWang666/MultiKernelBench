
#include "average_pooling_2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling2dCustomTilingData tiling;
    
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    uint32_t batchSize = inputShape->GetStorageShape().GetDim(0);
    uint32_t channels = inputShape->GetStorageShape().GetDim(1);
    uint32_t inputHeight = inputShape->GetStorageShape().GetDim(2);
    uint32_t inputWidth = inputShape->GetStorageShape().GetDim(3);
    
    const auto* attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetInt(0));
    uint32_t stride = *(attrs->GetInt(1));
    uint32_t padding = *(attrs->GetInt(2));
    
    uint32_t outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    uint32_t outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    
    uint32_t totalWork = batchSize * channels;
    uint32_t blockDim = totalWork < 32 ? totalWork : 32;
    
    context->SetBlockDim(blockDim);
    
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_inputHeight(inputHeight);
    tiling.set_inputWidth(inputWidth);
    tiling.set_outputHeight(outputHeight);
    tiling.set_outputWidth(outputWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    
    const auto* attrs = context->GetAttrs();
    int64_t kernelSize = *(attrs->GetInt(0));
    int64_t stride = *(attrs->GetInt(1));
    int64_t padding = *(attrs->GetInt(2));
    
    int64_t inH = x_shape->GetDim(2);
    int64_t inW = x_shape->GetDim(3);
    int64_t outH = (inH + 2 * padding - kernelSize) / stride + 1;
    int64_t outW = (inW + 2 * padding - kernelSize) / stride + 1;
    
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, outH);
    y_shape->SetDim(3, outW);
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
class AveragePooling2dCustom : public OpDef {
public:
    explicit AveragePooling2dCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("kernel_size")
            .Int();
        this->Attr("stride")
            .Int();
        this->Attr("padding")
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AveragePooling2dCustom);
}
