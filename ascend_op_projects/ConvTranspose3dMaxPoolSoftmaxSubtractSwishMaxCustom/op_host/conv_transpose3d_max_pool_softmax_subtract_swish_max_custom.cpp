
#include "conv_transpose3d_max_pool_softmax_subtract_swish_max_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 40;
const uint32_t TILE_ROWS = 240;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalElems = xShape->GetOriginShape().GetShapeSize();
    uint32_t totalRows = totalElems / 16;

    uint32_t blockDim = BLOCK_DIM;
    if (totalRows < blockDim) {
        blockDim = (totalRows == 0) ? 1 : totalRows;
    }
    uint32_t rowsPerCore = (totalRows + blockDim - 1) / blockDim;

    context->SetBlockDim(blockDim);
    tiling.set_totalRows(totalRows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_tileRows(TILE_ROWS);
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
    y_shape->SetDimNum(1);
    int64_t totalRows = x_shape->GetDim(0);
    y_shape->SetDim(0, totalRows);
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
class ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sub")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
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

OP_ADD(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom);
}
