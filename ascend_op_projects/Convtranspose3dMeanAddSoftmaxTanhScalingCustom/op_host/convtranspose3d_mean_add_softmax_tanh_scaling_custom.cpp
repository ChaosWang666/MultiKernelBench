
#include "convtranspose3d_mean_add_softmax_tanh_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose3dMeanAddSoftmaxTanhScalingCustomTilingData tiling;

    // x shape: (B, C, D, H, W)
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t B = x_shape->GetDim(0);
    uint32_t C = x_shape->GetDim(1);
    uint32_t D = x_shape->GetDim(2);
    uint32_t H = x_shape->GetDim(3);
    uint32_t W = x_shape->GetDim(4);

    // Get scaling factor from attrs
    const float* scalingPtr = context->GetAttrs()->GetAttrPointer<float>(0);
    float scalingFactor = 2.0f;
    if (scalingPtr) {
        scalingFactor = *scalingPtr;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(B);
    tiling.set_channels(C);
    tiling.set_depth(D);
    tiling.set_height(H);
    tiling.set_width(W);
    tiling.set_scalingFactor(scalingFactor);
    tiling.set_tileNum(TILE_NUM);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // workspace for intermediate results
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = B * C * H * W * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    // Output: (B, C, 1, H, W)
    *y_shape = *x_shape;
    y_shape->SetDim(2, 1);
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
class Convtranspose3dMeanAddSoftmaxTanhScalingCustom : public OpDef {
public:
    explicit Convtranspose3dMeanAddSoftmaxTanhScalingCustom(const char* name) : OpDef(name)
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
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("scaling_factor").AttrType(OPTIONAL).Float(2.0f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Convtranspose3dMeanAddSoftmaxTanhScalingCustom);
}
