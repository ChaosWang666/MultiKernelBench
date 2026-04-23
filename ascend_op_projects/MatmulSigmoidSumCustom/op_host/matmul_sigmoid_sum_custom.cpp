
#include "matmul_sigmoid_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t TILE_LEN = 4096;
static constexpr uint32_t ROWS_PER_CORE = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulSigmoidSumCustomTilingData tiling;
    const gert::Shape& inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = static_cast<uint32_t>(inputShape.GetDim(0));
    uint32_t hiddenSize = static_cast<uint32_t>(inputShape.GetDim(1));

    uint32_t rowsPerCore = ROWS_PER_CORE;
    uint32_t usedBlocks = (batchSize + rowsPerCore - 1) / rowsPerCore;

    uint32_t tileLength = TILE_LEN;
    uint32_t numTiles = hiddenSize / tileLength;

    context->SetBlockDim(usedBlocks);
    tiling.set_batchSize(batchSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_tileLength(tileLength);
    tiling.set_numTiles(numTiles);
    tiling.set_rowsPerCore(rowsPerCore);

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
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
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
class MatmulSigmoidSumCustom : public OpDef {
public:
    explicit MatmulSigmoidSumCustom(const char* name) : OpDef(name)
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

OP_ADD(MatmulSigmoidSumCustom);
}
