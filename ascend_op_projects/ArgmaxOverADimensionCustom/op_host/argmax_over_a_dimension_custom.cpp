
#include "argmax_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ArgmaxOverADimensionCustomTilingData tiling;
    
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    int32_t dimCount = inputShape->GetStorageShape().GetDimNum();
    
    // Get dim attribute
    const int64_t* dimPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    if (dim < 0) dim += dimCount;
    
    uint32_t outerSize = 1;
    uint32_t dimSize = 1;
    uint32_t innerSize = 1;
    
    for (int32_t i = 0; i < dim; i++) {
        outerSize *= inputShape->GetStorageShape().GetDim(i);
    }
    dimSize = inputShape->GetStorageShape().GetDim(dim);
    for (int32_t i = dim + 1; i < dimCount; i++) {
        innerSize *= inputShape->GetStorageShape().GetDim(i);
    }
    
    uint32_t totalLength = outerSize * innerSize;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dimSize(dimSize);
    tiling.set_outerSize(outerSize);
    tiling.set_innerSize(innerSize);
    tiling.set_tileNum(TILE_NUM);
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
    
    // Get dim from attrs
    const int64_t* dimPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    int32_t dimCount = x_shape->GetDimNum();
    if (dim < 0) dim += dimCount;
    
    // Output shape removes the dim dimension
    int32_t outIdx = 0;
    for (int32_t i = 0; i < dimCount; i++) {
        if (i != dim) {
            y_shape->SetDim(outIdx, x_shape->GetDim(i));
            outIdx++;
        }
    }
    y_shape->SetDimNum(dimCount - 1);
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // Argmax outputs int64
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ArgmaxOverADimensionCustom : public OpDef {
public:
    explicit ArgmaxOverADimensionCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("dim")
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ArgmaxOverADimensionCustom);
}
