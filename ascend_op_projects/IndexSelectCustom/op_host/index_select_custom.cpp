
#include "index_select_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    IndexSelectCustomTilingData tiling;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t dim = *(attrs->GetAttrPointer<uint32_t>(0));
    uint32_t numRows = *(attrs->GetAttrPointer<uint32_t>(1));
    uint32_t numCols = *(attrs->GetAttrPointer<uint32_t>(2));
    uint32_t numIndices = *(attrs->GetAttrPointer<uint32_t>(3));

    uint32_t blockDim = 32;
    context->SetBlockDim(blockDim);

    tiling.set_dim(dim);
    tiling.set_numRows(numRows);
    tiling.set_numCols(numCols);
    tiling.set_numIndices(numIndices);

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
    const gert::Shape* idx_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    uint32_t dim = *(attrs->GetAttrPointer<uint32_t>(0));

    int64_t ndim = x_shape->GetDimNum();
    for (int64_t i = 0; i < ndim; i++) {
        if (i == (int64_t)dim) {
            y_shape->SetDim(i, idx_shape->GetDim(0));
        } else {
            y_shape->SetDim(i, x_shape->GetDim(i));
        }
    }
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
class IndexSelectCustom : public OpDef {
public:
    explicit IndexSelectCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
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
        this->Attr("numRows")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("numCols")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("numIndices")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexSelectCustom);
}
