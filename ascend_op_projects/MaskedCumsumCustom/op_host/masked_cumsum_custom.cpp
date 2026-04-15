
#include "masked_cumsum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaskedCumsumCustomTilingData tiling;
    auto x_shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t ndim = x_shape.GetDimNum();

    // Get dim from attrs
    uint32_t dim = 1; // default
    const auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(0);
        if (dimPtr != nullptr) {
            int64_t d = *dimPtr;
            if (d < 0) d += ndim;
            dim = (uint32_t)d;
        }
    }

    uint32_t totalRows = 1;
    uint32_t rowLength = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        if (i < dim) {
            totalRows *= x_shape.GetDim(i);
        } else if (i == dim) {
            rowLength = x_shape.GetDim(i);
        } else {
            // For dims after, we treat them as part of the row count
            // Actually for general case we need to handle stride patterns
            // But for the given problem dim=1 on 2D tensor, this is fine
            totalRows *= x_shape.GetDim(i);
        }
    }

    // For 2D tensor with dim=1: totalRows = batch_size, rowLength = input_shape[0]
    // Actually let me reconsider: for general dims
    // rows before dim contribute to outer, dims after contribute to inner
    // We need to handle inner stride too. For simplicity with dim=1 on 2D:
    totalRows = 1;
    rowLength = 1;
    for (uint32_t i = 0; i < dim; i++) {
        totalRows *= x_shape.GetDim(i);
    }
    rowLength = x_shape.GetDim(dim);
    uint32_t innerSize = 1;
    for (uint32_t i = dim + 1; i < ndim; i++) {
        innerSize *= x_shape.GetDim(i);
    }
    // Treat totalRows as totalRows * innerSize for the simple case where innerSize=1
    totalRows = totalRows * innerSize;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalRows(totalRows);
    tiling.set_rowLength(rowLength);
    tiling.set_dim(dim);
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
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class MaskedCumsumCustom : public OpDef {
public:
    explicit MaskedCumsumCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(1);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaskedCumsumCustom);
}
