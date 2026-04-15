
#include "average_pooling_1d_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AveragePooling1dCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t channels = x_shape->GetDim(1);
    uint32_t inputLength = x_shape->GetDim(2);

    const uint64_t* attrs = context->GetAttrs();
    uint32_t kernelSize = *(const uint32_t*)(&attrs[0]);
    uint32_t stride = *(const uint32_t*)(&attrs[1]);
    uint32_t padding = *(const uint32_t*)(&attrs[2]);

    uint32_t outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;

    uint32_t totalChannels = batchSize * channels;
    uint32_t blockDim = 32;
    if (totalChannels < blockDim) {
        blockDim = totalChannels;
    }

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_inputLength(inputLength);
    tiling.set_outputLength(outputLength);
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

    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t channels = x_shape->GetDim(1);
    uint32_t inputLength = x_shape->GetDim(2);

    const auto* attrs = context->GetAttrs();
    int64_t kernelSize = *(const int64_t*)(attrs->GetAttrPointer(0));
    int64_t stride = *(const int64_t*)(attrs->GetAttrPointer(1));
    int64_t padding = *(const int64_t*)(attrs->GetAttrPointer(2));

    uint32_t outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;
    y_shape->SetDimNum(3);
    y_shape->SetDim(0, batchSize);
    y_shape->SetDim(1, channels);
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
class AveragePooling1dCustom : public OpDef {
public:
    explicit AveragePooling1dCustom(const char* name) : OpDef(name)
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
        this->Attr("kernelSize").AttrType(REQUIRED).Int();
        this->Attr("stride").AttrType(REQUIRED).Int();
        this->Attr("padding").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AveragePooling1dCustom);
}
