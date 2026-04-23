
#include "convtranspose2d_softmax_biasadd_scaling_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t TILE_SIZE = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose2dSoftmaxBiasaddScalingSigmoidCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = xShape.GetDim(0);
    uint32_t C = xShape.GetDim(1);
    uint32_t H = xShape.GetDim(2);
    uint32_t W = xShape.GetDim(3);
    uint32_t totalNC = N * C;
    uint32_t HW = H * W;

    auto attrs = context->GetAttrs();
    const float* scalingFactorPtr = attrs->GetAttrPointer<float>(0);
    float scalingFactor = (scalingFactorPtr != nullptr) ? *scalingFactorPtr : 1.0f;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalNC(totalNC);
    tiling.set_C(C);
    tiling.set_HW(HW);
    tiling.set_tileSize(TILE_SIZE);
    tiling.set_scalingFactor(scalingFactor);

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
class Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom : public OpDef {
public:
    explicit Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom(const char* name) : OpDef(name)
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
        this->Attr("scaling_factor").AttrType(OPTIONAL).Float(1.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom);
}
