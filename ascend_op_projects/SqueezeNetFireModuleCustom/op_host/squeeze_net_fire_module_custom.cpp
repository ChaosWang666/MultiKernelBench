
#include "squeeze_net_fire_module_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SqueezeNetFireModuleCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = input_shape->GetOriginShape().GetShapeVec();
    uint32_t batchSize = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t height = static_cast<uint32_t>(shape[2]);
    uint32_t width = static_cast<uint32_t>(shape[3]);

    uint32_t squeezeChannels = 6;
    uint32_t expand1x1Channels = 64;
    uint32_t expand3x3Channels = 64;
    uint32_t outChannels = expand1x1Channels + expand3x3Channels;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_squeezeChannels(squeezeChannels);
    tiling.set_expand1x1Channels(expand1x1Channels);
    tiling.set_expand3x3Channels(expand3x3Channels);
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
    const std::vector<int64_t>& input_shape = x1_shape->GetOriginShape().GetShapeVec();
    std::vector<int64_t> output_shape = {input_shape[0], 128, input_shape[2], input_shape[3]};
    *y_shape = gert::Shape(output_shape);
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
class SqueezeNetFireModuleCustom : public OpDef {
public:
    explicit SqueezeNetFireModuleCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SqueezeNetFireModuleCustom);
}
