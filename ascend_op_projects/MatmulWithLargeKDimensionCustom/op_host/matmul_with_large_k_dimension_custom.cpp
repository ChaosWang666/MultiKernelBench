
#include "matmul_with_large_k_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulWithLargeKDimensionCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetInputShape(1);
    
    uint32_t M = xShape->GetStorageShape().GetDim(0);
    uint32_t K = xShape->GetStorageShape().GetDim(1);
    uint32_t N = yShape->GetStorageShape().GetDim(1);
    
    // For large K, we tile along K dimension
    // Each tile processes a chunk of K
    uint32_t tileK = 1024; // process 1024 elements of K at a time
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_K(K);
    tiling.set_tileK(tileK);
    
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
    const gert::Shape* y_shape = context->GetInputShape(1);
    gert::Shape* z_shape = context->GetOutputShape(0);
    // Output shape is (M, N)
    z_shape->SetDimNum(2);
    z_shape->SetDim(0, x_shape->GetDim(0));
    z_shape->SetDim(1, y_shape->GetDim(1));
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
class MatmulWithLargeKDimensionCustom : public OpDef {
public:
    explicit MatmulWithLargeKDimensionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
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

OP_ADD(MatmulWithLargeKDimensionCustom);
}
