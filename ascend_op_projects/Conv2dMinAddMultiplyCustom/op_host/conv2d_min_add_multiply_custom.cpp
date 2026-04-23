
#include "conv2d_min_add_multiply_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 40;
const uint32_t TILE_SIZE = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dMinAddMultiplyCustomTilingData tiling;

    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = inputShape.GetDim(0);
    uint32_t C = inputShape.GetDim(1);
    uint32_t H = inputShape.GetDim(2);
    uint32_t W = inputShape.GetDim(3);

    uint32_t totalPairs = N * C;
    uint32_t spatialSize = H * W;

    auto attrs = context->GetAttrs();
    const float* constantValuePtr = attrs->GetAttrPointer<float>(0);
    const float* scalingFactorPtr = attrs->GetAttrPointer<float>(1);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalPairs(totalPairs);
    tiling.set_spatialSize(spatialSize);
    tiling.set_channels(C);
    tiling.set_tileSize(TILE_SIZE);
    tiling.set_constantValue(*constantValuePtr);
    tiling.set_scalingFactor(*scalingFactorPtr);

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
class Conv2dMinAddMultiplyCustom : public OpDef {
public:
    explicit Conv2dMinAddMultiplyCustom(const char* name) : OpDef(name)
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
        this->Attr("constant_value").AttrType(REQUIRED).Float();
        this->Attr("scaling_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dMinAddMultiplyCustom);
}
