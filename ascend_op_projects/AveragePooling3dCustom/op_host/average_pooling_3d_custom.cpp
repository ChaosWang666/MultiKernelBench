
#include "average_pooling_3d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling3dCustomTilingData tiling;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t stride = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t padding = *(attrs->GetAttrPointer<uint32_t>(2));
    uint32_t batchSize = *(attrs->GetAttrPointer<uint32_t>(3));
    uint32_t channels = *(attrs->GetAttrPointer<uint32_t>(4));
    uint32_t depth = *(attrs->GetAttrPointer<uint32_t>(5));
    uint32_t height = *(attrs->GetAttrPointer<uint32_t>(6));
    uint32_t width = *(attrs->GetAttrPointer<uint32_t>(7));

    uint32_t outDepth = (depth + 2 * padding - kernelSize) / stride + 1;
    uint32_t outHeight = (height + 2 * padding - kernelSize) / stride + 1;
    uint32_t outWidth = (width + 2 * padding - kernelSize) / stride + 1;

    uint32_t totalOutputElements = batchSize * channels * outDepth * outHeight * outWidth;

    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_outDepth(outDepth);
    tiling.set_outHeight(outHeight);
    tiling.set_outWidth(outWidth);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_totalOutputElements(totalOutputElements);

    uint32_t BLOCK_DIM = 32;
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t stride = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t padding = *(attrs->GetAttrPointer<uint32_t>(2));

    int64_t batch = x_shape->GetDim(0);
    int64_t ch = x_shape->GetDim(1);
    int64_t d = x_shape->GetDim(2);
    int64_t h = x_shape->GetDim(3);
    int64_t w = x_shape->GetDim(4);

    int64_t outD = (d + 2 * padding - kernelSize) / stride + 1;
    int64_t outH = (h + 2 * padding - kernelSize) / stride + 1;
    int64_t outW = (w + 2 * padding - kernelSize) / stride + 1;

    y_shape->SetDimNum(5);
    y_shape->SetDim(0, batch);
    y_shape->SetDim(1, ch);
    y_shape->SetDim(2, outD);
    y_shape->SetDim(3, outH);
    y_shape->SetDim(4, outW);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AveragePooling3dCustom : public OpDef {
public:
    explicit AveragePooling3dCustom(const char* name) : OpDef(name)
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
        this->Attr("kernel_size").AttrType(REQUIRED).Int();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();
        this->Attr("batch_size").AttrType(REQUIRED).Int();
        this->Attr("channels").AttrType(REQUIRED).Int();
        this->Attr("depth").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("width").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AveragePooling3dCustom);
}
