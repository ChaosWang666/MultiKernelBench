
#include "conv3d_group_norm_min_clamp_dropout_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dGroupNormMinClampDropoutCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t depth = static_cast<uint32_t>(shape[2]);
    uint32_t height = static_cast<uint32_t>(shape[3]);
    uint32_t width = static_cast<uint32_t>(shape[4]);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_depth(depth);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_groups(context->GetAttrInt("groups"));
    tiling.set_minValue(context->GetAttrFloat("min_value"));
    tiling.set_maxValue(context->GetAttrFloat("max_value"));
    tiling.set_dropoutP(context->GetAttrFloat("dropout_p"));
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
class Conv3dGroupNormMinClampDropoutCustom : public OpDef {
public:
    explicit Conv3dGroupNormMinClampDropoutCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Attr("in_channels").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("out_channels").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("kernel_size").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("groups").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("min_value").SetType(ATTR_TYPE_FLOAT).SetDefault(0.0f);
        this->Attr("max_value").SetType(ATTR_TYPE_FLOAT).SetDefault(1.0f);
        this->Attr("dropout_p").SetType(ATTR_TYPE_FLOAT).SetDefault(0.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dGroupNormMinClampDropoutCustom);
}
