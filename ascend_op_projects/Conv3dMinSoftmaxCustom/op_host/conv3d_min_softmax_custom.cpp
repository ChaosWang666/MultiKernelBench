
#include "conv3d_min_softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMinSoftmaxCustomTilingData tiling;

    auto attrs = context->GetAttrs();
    uint32_t dimD = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t channels = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t height = *(attrs->GetAttrPointer<uint32_t>(2));
    uint32_t width = *(attrs->GetAttrPointer<uint32_t>(3));
    uint32_t batchSize = *(attrs->GetAttrPointer<uint32_t>(4));

    uint32_t totalOutput = batchSize * channels * height * width;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_dimD(dimD);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_totalOutput(totalOutput);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    uint32_t channels = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t height = *(attrs->GetAttrPointer<uint32_t>(2));
    uint32_t width = *(attrs->GetAttrPointer<uint32_t>(3));
    uint32_t batchSize = *(attrs->GetAttrPointer<uint32_t>(4));

    *y_shape = gert::Shape({(int64_t)batchSize, (int64_t)channels, (int64_t)height, (int64_t)width});
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
class Conv3dMinSoftmaxCustom : public OpDef {
public:
    explicit Conv3dMinSoftmaxCustom(const char* name) : OpDef(name)
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
        this->Attr("dimD").AttrType(REQUIRED).Int();
        this->Attr("channels").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("batchSize").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dMinSoftmaxCustom);
}
