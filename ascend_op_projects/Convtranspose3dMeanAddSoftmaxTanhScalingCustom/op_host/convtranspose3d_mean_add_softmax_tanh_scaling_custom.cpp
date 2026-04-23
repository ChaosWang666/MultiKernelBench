
#include "convtranspose3d_mean_add_softmax_tanh_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 20;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose3dMeanAddSoftmaxTanhScalingCustomTilingData tiling;
    const gert::Shape& xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t B = xShape.GetDim(0);
    uint32_t C = xShape.GetDim(1);
    uint32_t D = xShape.GetDim(2);
    uint32_t H = xShape.GetDim(3);
    uint32_t W = xShape.GetDim(4);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_D(D);
    tiling.set_H(H);
    tiling.set_W(W);

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
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(5);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(1));
    yShape->SetDim(2, 1);
    yShape->SetDim(3, xShape->GetDim(3));
    yShape->SetDim(4, xShape->GetDim(4));
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
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Convtranspose3dMeanAddSoftmaxTanhScalingCustom);
}
