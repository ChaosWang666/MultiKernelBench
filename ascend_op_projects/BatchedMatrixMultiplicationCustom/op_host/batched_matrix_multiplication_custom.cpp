
#include "batched_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BatchedMatrixMultiplicationCustomTilingData tiling;
    
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    
    uint32_t batchSize = aShape->GetStorageShape().GetDim(0);
    uint32_t M = aShape->GetStorageShape().GetDim(1);
    uint32_t K = aShape->GetStorageShape().GetDim(2);
    uint32_t N = bShape->GetStorageShape().GetDim(2);
    
    // Use batch_size as block dim so each core handles one or more batches
    uint32_t blockDim = batchSize;
    if (blockDim > 32) blockDim = 32;
    
    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_M(M);
    tiling.set_K(K);
    tiling.set_N(N);
    
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
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    gert::Shape* cShape = context->GetOutputShape(0);
    
    cShape->SetDimNum(3);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, aShape->GetDim(1));
    cShape->SetDim(2, bShape->GetDim(2));
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
class BatchedMatrixMultiplicationCustom : public OpDef {
public:
    explicit BatchedMatrixMultiplicationCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("c")
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

OP_ADD(BatchedMatrixMultiplicationCustom);
}
