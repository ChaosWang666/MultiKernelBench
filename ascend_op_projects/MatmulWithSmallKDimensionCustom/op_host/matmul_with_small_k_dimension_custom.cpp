
#include "matmul_with_small_k_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulWithSmallKDimensionCustomTilingData tiling;
    
    const gert::StorageShape* aShape = context->GetInputShape(0);
    const gert::StorageShape* bShape = context->GetInputShape(1);
    
    uint32_t M = aShape->GetOriginShape().GetDim(0);
    uint32_t K = aShape->GetOriginShape().GetDim(1);
    uint32_t N = bShape->GetOriginShape().GetDim(1);
    
    // Each block processes a chunk of rows of the output
    // We tile along M and N dimensions
    uint32_t tileM = (M + BLOCK_DIM - 1) / BLOCK_DIM; // rows per block
    
    // For N, we process in tiles that fit in local memory
    // With K=64, each row of A is 64 floats = 256 bytes
    // We want to tile N so that B tile (K x tileN) and partial results fit
    // Local memory ~256KB per buffer, so tileN can be generous
    uint32_t tileN = 128; // process 128 columns of N at a time
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);
    tiling.set_tileM(tileM);
    tiling.set_tileN(tileN);
    
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
    cShape->SetDimNum(2);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, bShape->GetDim(1));
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
class MatmulWithSmallKDimensionCustom : public OpDef {
public:
    explicit MatmulWithSmallKDimensionCustom(const char* name) : OpDef(name)
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

OP_ADD(MatmulWithSmallKDimensionCustom);
}
