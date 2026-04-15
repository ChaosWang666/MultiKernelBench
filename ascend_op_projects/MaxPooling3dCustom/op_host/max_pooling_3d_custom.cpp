
#include "max_pooling_3d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxPooling3dCustomTilingData tiling;

    const auto attrs = context->GetAttrs();
    uint32_t kernelSize = *(attrs->GetInt(0));
    uint32_t stride = *(attrs->GetInt(1));
    uint32_t padding = *(attrs->GetInt(2));
    uint32_t batchSize = *(attrs->GetInt(3));
    uint32_t channels = *(attrs->GetInt(4));
    uint32_t dim1 = *(attrs->GetInt(5));
    uint32_t dim2 = *(attrs->GetInt(6));
    uint32_t dim3 = *(attrs->GetInt(7));
    uint32_t outDim1 = *(attrs->GetInt(8));
    uint32_t outDim2 = *(attrs->GetInt(9));
    uint32_t outDim3 = *(attrs->GetInt(10));

    uint32_t totalOutputElements = batchSize * channels * outDim1 * outDim2 * outDim3;

    uint32_t blockDim = 32;
    if (totalOutputElements < blockDim) {
        blockDim = totalOutputElements;
    }

    context->SetBlockDim(blockDim);

    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_dim3(dim3);
    tiling.set_outDim1(outDim1);
    tiling.set_outDim2(outDim2);
    tiling.set_outDim3(outDim3);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_padding(padding);
    tiling.set_totalOutputElements(totalOutputElements);
    tiling.set_blockDim(blockDim);

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
    // Output shape is set by the attrs
    // We just need to read attrs and set shape
    const auto attrs = context->GetAttrs();
    uint32_t batchSize = *(attrs->GetInt(3));
    uint32_t channels = *(attrs->GetInt(4));
    uint32_t outDim1 = *(attrs->GetInt(8));
    uint32_t outDim2 = *(attrs->GetInt(9));
    uint32_t outDim3 = *(attrs->GetInt(10));
    
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, batchSize);
    y_shape->SetDim(1, channels);
    y_shape->SetDim(2, outDim1);
    y_shape->SetDim(3, outDim2);
    y_shape->SetDim(4, outDim3);
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
class MaxPooling3dCustom : public OpDef {
public:
    explicit MaxPooling3dCustom(const char* name) : OpDef(name)
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
        this->Attr("dim1").AttrType(REQUIRED).Int();
        this->Attr("dim2").AttrType(REQUIRED).Int();
        this->Attr("dim3").AttrType(REQUIRED).Int();
        this->Attr("out_dim1").AttrType(REQUIRED).Int();
        this->Attr("out_dim2").AttrType(REQUIRED).Int();
        this->Attr("out_dim3").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxPooling3dCustom);
}
