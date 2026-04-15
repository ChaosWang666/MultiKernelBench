
#include "sum_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SumReductionOverADimensionCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    
    const auto *attrs = context->GetAttrs();
    uint32_t dim = *(attrs->GetAttrPointer<int32_t>(0));
    uint32_t s0 = *(attrs->GetAttrPointer<int32_t>(1));
    uint32_t s1 = *(attrs->GetAttrPointer<int32_t>(2));
    uint32_t s2 = *(attrs->GetAttrPointer<int32_t>(3));
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dim(dim);
    tiling.set_shape0(s0);
    tiling.set_shape1(s1);
    tiling.set_shape2(s2);
    tiling.set_tileNum(TILE_NUM);
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
class SumReductionOverADimensionCustom : public OpDef {
public:
    explicit SumReductionOverADimensionCustom(const char* name) : OpDef(name)
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
        this->Attr("dim")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("shape0")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("shape1")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("shape2")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SumReductionOverADimensionCustom);
}
