
#include "conv2d_add_scale_sigmoid_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 40;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dAddScaleSigmoidGroupNormCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = xShape.GetDim(0);
    uint32_t channels = xShape.GetDim(1);
    uint32_t height = xShape.GetDim(2);
    uint32_t width = xShape.GetDim(3);
    uint32_t hw = height * width;

    auto attrs = context->GetAttrs();
    const int64_t* num_groups_ptr = attrs->GetAttrPointer<int64_t>(0);
    const float* eps_ptr = attrs->GetAttrPointer<float>(1);
    uint32_t numGroups = static_cast<uint32_t>(*num_groups_ptr);
    float eps = *eps_ptr;
    uint32_t channelsPerGroup = channels / numGroups;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_hw(hw);
    tiling.set_numGroups(numGroups);
    tiling.set_channelsPerGroup(channelsPerGroup);
    tiling.set_eps(eps);
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
class Conv2dAddScaleSigmoidGroupNormCustom : public OpDef {
public:
    explicit Conv2dAddScaleSigmoidGroupNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dAddScaleSigmoidGroupNormCustom);
}
