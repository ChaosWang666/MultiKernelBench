
#include "conv_transpose3d_softmax_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t ROWS_PER_TILE = 64;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dSoftmaxSigmoidCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t dimNum = shape.GetDimNum();
    uint32_t cols = static_cast<uint32_t>(shape.GetDim(dimNum - 1));
    uint32_t totalRows = 1;
    for (uint32_t i = 0; i < dimNum - 1; i++) {
        totalRows *= static_cast<uint32_t>(shape.GetDim(i));
    }

    uint32_t numBlocks = BLOCK_DIM;
    if (totalRows < numBlocks) {
        numBlocks = totalRows == 0 ? 1 : totalRows;
    }
    uint32_t rowsPerCore = (totalRows + numBlocks - 1) / numBlocks;

    context->SetBlockDim(numBlocks);
    tiling.set_totalRows(totalRows);
    tiling.set_cols(cols);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_rowsPerTile(ROWS_PER_TILE);

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
class ConvTranspose3dSoftmaxSigmoidCustom : public OpDef {
public:
    explicit ConvTranspose3dSoftmaxSigmoidCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dSoftmaxSigmoidCustom);
}
