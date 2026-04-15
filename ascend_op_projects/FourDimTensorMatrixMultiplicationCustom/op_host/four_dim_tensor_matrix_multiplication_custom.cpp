
#include "four_dim_tensor_matrix_multiplication_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    FourDimTensorMatrixMultiplicationCustomTilingData tiling;
    
    const gert::Shape* aShape = context->GetInputShape(0);
    const gert::Shape* bShape = context->GetInputShape(1);
    
    uint32_t b = aShape->GetDim(0);
    uint32_t i = aShape->GetDim(1);
    uint32_t j = aShape->GetDim(2);
    uint32_t l = aShape->GetDim(3);
    uint32_t k = bShape->GetDim(1);
    
    uint32_t batchSize = b;
    uint32_t M = i * j;
    uint32_t K = l;
    uint32_t N = k;
    
    context->SetBlockDim(BLOCK_DIM);
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
    
    cShape->SetDimNum(4);
    cShape->SetDim(0, aShape->GetDim(0));
    cShape->SetDim(1, aShape->GetDim(1));
    cShape->SetDim(2, aShape->GetDim(2));
    cShape->SetDim(3, bShape->GetDim(1));
    
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
class FourDimTensorMatrixMultiplicationCustom : public OpDef {
public:
    explicit FourDimTensorMatrixMultiplicationCustom(const char* name) : OpDef(name)
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

OP_ADD(FourDimTensorMatrixMultiplicationCustom);
}
