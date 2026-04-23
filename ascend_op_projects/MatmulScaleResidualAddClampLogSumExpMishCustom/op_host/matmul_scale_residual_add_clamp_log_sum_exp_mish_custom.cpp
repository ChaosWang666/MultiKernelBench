
#include "matmul_scale_residual_add_clamp_log_sum_exp_mish_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulScaleResidualAddClampLogSumExpMishCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = xShape.GetDim(0);
    uint32_t hiddenSize = xShape.GetDim(1);

    auto attrs = context->GetAttrs();
    const float* scaleFactor = attrs->GetAttrPointer<float>(0);
    const float* clampMin = attrs->GetAttrPointer<float>(1);
    const float* clampMax = attrs->GetAttrPointer<float>(2);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_scaleFactor(*scaleFactor);
    tiling.set_clampMin(*clampMin);
    tiling.set_clampMax(*clampMax);
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
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
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
class MatmulScaleResidualAddClampLogSumExpMishCustom : public OpDef {
public:
    explicit MatmulScaleResidualAddClampLogSumExpMishCustom(const char* name) : OpDef(name)
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

        this->Attr("scale_factor").AttrType(REQUIRED).Float();
        this->Attr("clamp_min").AttrType(REQUIRED).Float();
        this->Attr("clamp_max").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulScaleResidualAddClampLogSumExpMishCustom);
}
