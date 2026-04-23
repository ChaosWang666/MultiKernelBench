
#include "conv2d_multiply_leaky_relu_gelu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 40;
const uint32_t DEFAULT_TILE_LENGTH = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dMultiplyLeakyReluGeluCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();

    uint32_t N = xShape.GetDim(0);
    uint32_t C = xShape.GetDim(1);
    uint32_t H = xShape.GetDim(2);
    uint32_t W = xShape.GetDim(3);

    uint32_t totalChannels = N * C;
    uint32_t channelSize = H * W;
    uint32_t channels = C;
    uint32_t tileLength = DEFAULT_TILE_LENGTH;
    if (channelSize < tileLength) {
        tileLength = channelSize;
    }

    uint32_t usedBlocks = BLOCK_DIM;
    if (totalChannels < usedBlocks) {
        usedBlocks = totalChannels;
    }

    context->SetBlockDim(usedBlocks);
    tiling.set_totalChannels(totalChannels);
    tiling.set_channelSize(channelSize);
    tiling.set_channels(channels);
    tiling.set_tileLength(tileLength);

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
class Conv2dMultiplyLeakyReluGeluCustom : public OpDef {
public:
    explicit Conv2dMultiplyLeakyReluGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("multiplier")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dMultiplyLeakyReluGeluCustom);
}
