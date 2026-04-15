
#include "cumsum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CumsumCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t totalLength = x_shape->GetOriginShape().GetShapeSize();
    uint32_t numDims = x_shape->GetOriginShape().GetDimNum();

    // Get the dim attribute
    const auto* attrs = context->GetAttrs();
    int64_t dim = *(attrs->GetAttrPointer<int64_t>(0));
    if (dim < 0) {
        dim += numDims;
    }

    uint32_t dimSize = x_shape->GetOriginShape().GetDim(dim);

    uint32_t outerSize = 1;
    for (uint32_t i = 0; i < (uint32_t)dim; i++) {
        outerSize *= x_shape->GetOriginShape().GetDim(i);
    }

    uint32_t innerSize = 1;
    for (uint32_t i = (uint32_t)dim + 1; i < numDims; i++) {
        innerSize *= x_shape->GetOriginShape().GetDim(i);
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dim((uint32_t)dim);
    tiling.set_dimSize(dimSize);
    tiling.set_outerSize(outerSize);
    tiling.set_innerSize(innerSize);
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
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class CumsumCustom : public OpDef {
public:
    explicit CumsumCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CumsumCustom);
}
