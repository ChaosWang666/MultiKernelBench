
#include "googlenet_inception_module_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GooglenetInceptionModuleCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_inChannels(context->GetAttrInt("in_channels"));
    tiling.set_out1x1(context->GetAttrInt("out_1x1"));
    tiling.set_reduce3x3(context->GetAttrInt("reduce_3x3"));
    tiling.set_out3x3(context->GetAttrInt("out_3x3"));
    tiling.set_reduce5x5(context->GetAttrInt("reduce_5x5"));
    tiling.set_out5x5(context->GetAttrInt("out_5x5"));
    tiling.set_poolProj(context->GetAttrInt("pool_proj"));
    tiling.set_batchSize(context->GetInputShape(0)->GetOriginShape().GetDim(0));
    tiling.set_height(context->GetInputShape(0)->GetOriginShape().GetDim(2));
    tiling.set_width(context->GetInputShape(0)->GetOriginShape().GetDim(3));
    tiling.set_totalLength(totalLength);
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
    uint32_t inChannels = context->GetAttrInt("in_channels");
    uint32_t out1x1 = context->GetAttrInt("out_1x1");
    uint32_t out3x3 = context->GetAttrInt("out_3x3");
    uint32_t out5x5 = context->GetAttrInt("out_5x5");
    uint32_t poolProj = context->GetAttrInt("pool_proj");
    uint32_t outChannels = out1x1 + out3x3 + out5x5 + poolProj;
    *y_shape = gert::Shape({x1_shape->GetOriginShape().GetDim(0), outChannels, x1_shape->GetOriginShape().GetDim(2), x1_shape->GetOriginShape().GetDim(3)});
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
class GooglenetInceptionModuleCustom : public OpDef {
public:
    explicit GooglenetInceptionModuleCustom(const char* name) : OpDef(name)
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
        this->Attr("in_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_1x1").SetType(INT).SetParamType(REQUIRED);
        this->Attr("reduce_3x3").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_3x3").SetType(INT).SetParamType(REQUIRED);
        this->Attr("reduce_5x5").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_5x5").SetType(INT).SetParamType(REQUIRED);
        this->Attr("pool_proj").SetType(INT).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GooglenetInceptionModuleCustom);
}
