
#include "l1_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    L1NormCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t dim = xShape->GetStorageShape().GetDim(1);

    // Each block processes some rows
    uint32_t BLOCK_DIM = 32;
    // tileNum controls how many tiles we split each row's dim into
    // We want tileLength to be a multiple of 32 (256 bytes / 4 bytes per float = 64, but 32 is min alignment)
    // Pick tileNum so that tileLength = dim / tileNum is reasonable
    uint32_t tileNum = 1;
    // We want each tile to fit in UB. UB is ~256KB, we need 2 buffers for input + 1 for output + 1 temp
    // So max tile size ~ 64KB / 4 = 16384 floats
    uint32_t maxTileSize = 16384;
    if (dim > maxTileSize) {
        tileNum = (dim + maxTileSize - 1) / maxTileSize;
        // Round up tileNum so tileLength is aligned to 32
        // Actually let's find tileNum such that dim / tileNum is aligned to 8 (32 bytes)
        // Try to find a good tileNum
        uint32_t tileLength = (dim + tileNum - 1) / tileNum;
        // Align tileLength up to 8
        tileLength = ((tileLength + 7) / 8) * 8;
        tileNum = (dim + tileLength - 1) / tileLength;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_dim(dim);
    tiling.set_tileNum(tileNum);
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
class L1NormCustom : public OpDef {
public:
    explicit L1NormCustom(const char* name) : OpDef(name)
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

OP_ADD(L1NormCustom);
}
