
#include "cumprod_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CumprodCustomTilingData tiling;
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int32_t ndim = x_shape->GetStorageShape().GetDimNum();
    uint32_t totalLength = x_shape->GetStorageShape().GetShapeSize();

    // Get dim attribute
    const int64_t* dimPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    if (dim < 0) dim += ndim;

    uint32_t dimSize = x_shape->GetStorageShape().GetDim(dim);

    uint32_t outerSize = 1;
    for (int32_t i = 0; i < dim; i++) {
        outerSize *= x_shape->GetStorageShape().GetDim(i);
    }

    uint32_t innerSize = 1;
    for (int32_t i = dim + 1; i < ndim; i++) {
        innerSize *= x_shape->GetStorageShape().GetDim(i);
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
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class CumprodCustom : public OpDef {
public:
    explicit CumprodCustom(const char* name) : OpDef(name)
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

OP_ADD(CumprodCustom);
}
