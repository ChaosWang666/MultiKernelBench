
#include "conv3d_scaling_tanh_multiply_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static const uint32_t MAX_BLOCK_DIM = 40;
static const uint32_t DEFAULT_TILE_LENGTH = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dScalingTanhMultiplySigmoidCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t B = xShape.GetDim(0);
    uint32_t C = xShape.GetDim(1);
    uint32_t D = xShape.GetDim(2);
    uint32_t H = xShape.GetDim(3);
    uint32_t W = xShape.GetDim(4);

    uint32_t totalChannels = B * C;
    uint32_t perChannelSize = D * H * W;

    uint32_t blockDim = (totalChannels < MAX_BLOCK_DIM) ? totalChannels : MAX_BLOCK_DIM;
    if (blockDim == 0) {
        blockDim = 1;
    }
    uint32_t channelsPerCore = totalChannels / blockDim;
    uint32_t tailChannels = totalChannels % blockDim;
    uint32_t tileLength = DEFAULT_TILE_LENGTH;
    if (tileLength > perChannelSize) {
        tileLength = perChannelSize;
    }

    context->SetBlockDim(blockDim);
    tiling.set_totalChannels(totalChannels);
    tiling.set_perChannelSize(perChannelSize);
    tiling.set_channels(C);
    tiling.set_channelsPerCore(channelsPerCore);
    tiling.set_tailChannels(tailChannels);
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
class Conv3dScalingTanhMultiplySigmoidCustom : public OpDef {
public:
    explicit Conv3dScalingTanhMultiplySigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(Conv3dScalingTanhMultiplySigmoidCustom);
}
