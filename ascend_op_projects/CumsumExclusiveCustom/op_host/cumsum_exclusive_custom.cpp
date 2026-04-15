
#include "cumsum_exclusive_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CumsumExclusiveCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int32_t dimCount = xShape->GetStorageShape().GetDimNum();

    // Get the dim attribute
    const auto* attrs = context->GetAttrs();
    int32_t dim = *(attrs->GetAttrPointer<int32_t>(0));
    if (dim < 0) dim += dimCount;

    uint32_t totalLength = 1;
    for (int i = 0; i < dimCount; i++) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }

    uint32_t dimSize = xShape->GetStorageShape().GetDim(dim);

    uint32_t outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= xShape->GetStorageShape().GetDim(i);
    }

    uint32_t innerSize = 1;
    for (int i = dim + 1; i < dimCount; i++) {
        innerSize *= xShape->GetStorageShape().GetDim(i);
    }

    // Use outerSize * innerSize as block dim, but cap it
    uint32_t numLines = outerSize * innerSize;
    uint32_t blockDim = numLines < 32 ? numLines : 32;

    context->SetBlockDim(blockDim);
    tiling.set_totalLength(totalLength);
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
class CumsumExclusiveCustom : public OpDef {
public:
    explicit CumsumExclusiveCustom(const char* name) : OpDef(name)
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

OP_ADD(CumsumExclusiveCustom);
}
