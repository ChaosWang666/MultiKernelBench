
#include "take_along_dim_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TakeAlongDimCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* idxShape = context->GetInputShape(1);
    uint32_t numRows = xShape->GetStorageShape().GetDim(0);
    uint32_t srcCols = xShape->GetStorageShape().GetDim(1);
    uint32_t idxCols = idxShape->GetStorageShape().GetDim(1);

    // Use one block per row, up to 32 blocks
    uint32_t blockDim = numRows < 32 ? numRows : 32;
    context->SetBlockDim(blockDim);

    tiling.set_numRows(numRows);
    tiling.set_srcCols(srcCols);
    tiling.set_idxCols(idxCols);
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
    const gert::Shape* idxShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *idxShape;
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
class TakeAlongDimCustom : public OpDef {
public:
    explicit TakeAlongDimCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TakeAlongDimCustom);
}
