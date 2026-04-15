
#include "adam_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 2048;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AdamCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);

    const auto* attrs = context->GetAttrs();
    float beta1 = *(attrs->GetAttrPointer<float>(0));
    float beta2 = *(attrs->GetAttrPointer<float>(1));
    float lr = *(attrs->GetAttrPointer<float>(2));
    float eps = *(attrs->GetAttrPointer<float>(3));
    int step = *(attrs->GetAttrPointer<int>(4));

    float beta1Power = std::pow(beta1, step);
    float beta2Power = std::pow(beta2, step);
    float beta1CorrInv = 1.0f / (1.0f - beta1Power);
    float beta2CorrInv = 1.0f / (1.0f - beta2Power);

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.set_beta1(beta1);
    tiling.set_beta2(beta2);
    tiling.set_lr(lr);
    tiling.set_eps(eps);
    tiling.set_beta1CorrInv(beta1CorrInv);
    tiling.set_beta2CorrInv(beta2CorrInv);

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
class AdamCustom : public OpDef {
public:
    explicit AdamCustom(const char* name) : OpDef(name)
    {
        this->Input("param")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("m")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("param_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("beta1").AttrType(REQUIRED).Float();
        this->Attr("beta2").AttrType(REQUIRED).Float();
        this->Attr("lr").AttrType(REQUIRED).Float();
        this->Attr("eps").AttrType(REQUIRED).Float();
        this->Attr("step").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AdamCustom);
}
