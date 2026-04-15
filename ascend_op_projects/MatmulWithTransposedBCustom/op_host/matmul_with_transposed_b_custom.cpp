
#include "matmul_with_transposed_b_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulWithTransposedBCustomTilingData tiling;
    
    const gert::StorageShape* a_shape = context->GetInputShape(0);
    const gert::StorageShape* b_shape = context->GetInputShape(1);
    
    uint32_t M = a_shape->GetStorageShape().GetDim(0);
    uint32_t K = a_shape->GetStorageShape().GetDim(1);
    uint32_t N = b_shape->GetStorageShape().GetDim(0);
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);
    
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
    const gert::Shape* a_shape = context->GetInputShape(0);
    const gert::Shape* b_shape = context->GetInputShape(1);
    gert::Shape* c_shape = context->GetOutputShape(0);
    
    c_shape->SetDimNum(2);
    c_shape->SetDim(0, a_shape->GetDim(0));
    c_shape->SetDim(1, b_shape->GetDim(0));
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
class MatmulWithTransposedBCustom : public OpDef {
public:
    explicit MatmulWithTransposedBCustom(const char* name) : OpDef(name)
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

OP_ADD(MatmulWithTransposedBCustom);
}
