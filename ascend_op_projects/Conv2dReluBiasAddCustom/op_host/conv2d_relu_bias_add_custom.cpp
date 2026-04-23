
#include "conv2d_relu_bias_add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 40;
const uint32_t TILE_LENGTH = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dReluBiasAddCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(xShape.GetDim(3));

    uint32_t totalGroups = N * C;
    uint32_t elementsPerGroup = H * W;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalGroups(totalGroups);
    tiling.set_elementsPerGroup(elementsPerGroup);
    tiling.set_outChannels(C);
    tiling.set_tileLength(TILE_LENGTH);

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
class Conv2dReluBiasAddCustom : public OpDef {
public:
    explicit Conv2dReluBiasAddCustom(const char* name) : OpDef(name)
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

OP_ADD(Conv2dReluBiasAddCustom);
}
