
#include "argmin_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgminOverADimensionCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int32_t dimCount = xShape->GetStorageShape().GetDimNum();

    const auto* attrs = context->GetAttrs();
    int64_t dim = *(attrs->GetAttrPointer<int64_t>(0));
    if (dim < 0) dim += dimCount;

    uint32_t dimBefore = 1;
    uint32_t dimSize = 1;
    uint32_t dimAfter = 1;
    uint32_t totalLength = 1;

    for (int32_t i = 0; i < dimCount; i++) {
        uint32_t s = xShape->GetStorageShape().GetDim(i);
        totalLength *= s;
        if (i < dim) dimBefore *= s;
        else if (i == dim) dimSize = s;
        else dimAfter *= s;
    }

    uint32_t numOutputElements = dimBefore * dimAfter;
    uint32_t blockDim = 32;
    if (numOutputElements < blockDim) blockDim = numOutputElements;

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
    tiling.set_dimBefore(dimBefore);
    tiling.set_dimSize(dimSize);
    tiling.set_dimAfter(dimAfter);
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
    int32_t dimCount = x_shape->GetDimNum();

    const auto* attrs = context->GetAttrs();
    int64_t dim = *(attrs->GetAttrPointer<int64_t>(0));
    if (dim < 0) dim += dimCount;

    // Output shape removes the dim dimension
    int32_t outIdx = 0;
    for (int32_t i = 0; i < dimCount; i++) {
        if (i != dim) {
            y_shape->SetDim(outIdx, x_shape->GetDim(i));
            outIdx++;
        }
    }
    y_shape->SetDimNum(outIdx);
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ArgminOverADimensionCustom : public OpDef {
public:
    explicit ArgminOverADimensionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
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

OP_ADD(ArgminOverADimensionCustom);
}
