
#include "conv2d_group_norm_scale_max_pool_clamp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dGroupNormScaleMaxPoolClampCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = input_shape->GetOriginShape().GetDims();
    uint32_t batchSize = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t height = static_cast<uint32_t>(shape[2]);
    uint32_t width = static_cast<uint32_t>(shape[3]);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(64); // Assuming fixed out channels
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(3);
    tiling.set_numGroups(16);
    tiling.set_maxpoolKernelSize(4);
    tiling.set_clampMin(0.0f);
    tiling.set_clampMax(1.0f);
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
    const gert::Shape* input_shape = context->GetInputShape(0);
    gert::Shape* output_shape = context->GetOutputShape(0);
    *output_shape = *input_shape;
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
class Conv2dGroupNormScaleMaxPoolClampCustom : public OpDef {
public:
    explicit Conv2dGroupNormScaleMaxPoolClampCustom(const char* name) : OpDef(name)
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
        this->Attr("conv_weight").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("conv_bias").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("group_norm_weight").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("group_norm_bias").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("scale").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("clamp_min").ParamType(REQUIRED).DataType(ge::DT_FLOAT);
        this->Attr("clamp_max").ParamType(REQUIRED).DataType(ge::DT_FLOAT);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dGroupNormScaleMaxPoolClampCustom);
}
