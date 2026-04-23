
#include "conv3d_min_softmax_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMinSoftmaxCustomTilingData tiling;
    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = (uint32_t)inputShape.GetDim(0);
    uint32_t C = (uint32_t)inputShape.GetDim(1);
    uint32_t D = (uint32_t)inputShape.GetDim(2);
    uint32_t H = (uint32_t)inputShape.GetDim(3);
    uint32_t W = (uint32_t)inputShape.GetDim(4);

    uint32_t totalSlices = N * H;
    uint32_t blockDim = BLOCK_DIM;
    if (totalSlices < blockDim) {
        blockDim = totalSlices;
    }
    uint32_t slicesPerCore = (totalSlices + blockDim - 1) / blockDim;

    context->SetBlockDim(blockDim);
    tiling.set_N(N);
    tiling.set_C(C);
    tiling.set_D(D);
    tiling.set_H(H);
    tiling.set_W(W);
    tiling.set_totalSlices(totalSlices);
    tiling.set_slicesPerCore(slicesPerCore);

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
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, x_shape->GetDim(3));
    y_shape->SetDim(3, x_shape->GetDim(4));
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
class Conv3dMinSoftmaxCustom : public OpDef {
public:
    explicit Conv3dMinSoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(Conv3dMinSoftmaxCustom);
}
