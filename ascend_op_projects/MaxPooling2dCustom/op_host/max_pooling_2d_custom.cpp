
#include "max_pooling_2d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxPooling2dCustomTilingData tiling;
    
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t stride = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t padding = *(attrs->GetAttrPointer<uint32_t>(2));
    uint32_t batchSize = *(attrs->GetAttrPointer<uint32_t>(3));
    uint32_t channels = *(attrs->GetAttrPointer<uint32_t>(4));
    uint32_t height = *(attrs->GetAttrPointer<uint32_t>(5));
    uint32_t width = *(attrs->GetAttrPointer<uint32_t>(6));
    
    uint32_t outHeight = (height + 2 * padding - kernelSize) / stride + 1;
    uint32_t outWidth = (width + 2 * padding - kernelSize) / stride + 1;
    
    uint32_t totalPlanes = batchSize * channels;
    uint32_t blockDim = totalPlanes < 32 ? totalPlanes : 32;
    
    context->SetBlockDim(blockDim);
    
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
    
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
    
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t stride = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t padding = *(attrs->GetAttrPointer<uint32_t>(2));
    
    int64_t N = x_shape->GetDim(0);
    int64_t C = x_shape->GetDim(1);
    int64_t H = x_shape->GetDim(2);
    int64_t W = x_shape->GetDim(3);
    
    int64_t outH = (H + 2 * padding - kernelSize) / stride + 1;
    int64_t outW = (W + 2 * padding - kernelSize) / stride + 1;
    
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, N);
    y_shape->SetDim(1, C);
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
class MaxPooling2dCustom : public OpDef {
public:
    explicit MaxPooling2dCustom(const char* name) : OpDef(name)
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
            .AttrType(REQUIRED)
            .Int();
        this->Attr("stride")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("padding")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("channels")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("height")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("width")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxPooling2dCustom);
}
