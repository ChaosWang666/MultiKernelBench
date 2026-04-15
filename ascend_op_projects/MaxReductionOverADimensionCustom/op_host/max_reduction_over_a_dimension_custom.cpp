
#include "max_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MaxReductionOverADimensionCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int32_t dimCount = xShape->GetStorageShape().GetDimNum();
    
    // Get the dim attribute
    const int64_t* dimPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    if (dim < 0) dim += dimCount;
    
    uint32_t dimBefore = 1;
    uint32_t dimSize = 1;
    uint32_t dimAfter = 1;
    
    for (int32_t i = 0; i < dimCount; i++) {
        int64_t s = xShape->GetStorageShape().GetDim(i);
        if (i < dim) dimBefore *= (uint32_t)s;
        else if (i == dim) dimSize = (uint32_t)s;
        else dimAfter *= (uint32_t)s;
    }
    
    uint32_t totalLength = dimBefore * dimAfter; // output size
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dimBefore(dimBefore);
    tiling.set_dimSize(dimSize);
    tiling.set_dimAfter(dimAfter);
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
    
    // Get dim attribute
    const int64_t* dimPtr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    int32_t dimCount = x_shape->GetDimNum();
    if (dim < 0) dim += dimCount;
    
    // Output shape: remove the dim-th dimension
    int idx = 0;
    for (int32_t i = 0; i < dimCount; i++) {
        if (i != dim) {
            y_shape->SetDim(idx++, x_shape->GetDim(i));
        }
    }
    // Set the number of dims
    // We need to handle this carefully - the output has dimCount-1 dimensions
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
class MaxReductionOverADimensionCustom : public OpDef {
public:
    explicit MaxReductionOverADimensionCustom(const char* name) : OpDef(name)
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
        this->Attr("dim")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MaxReductionOverADimensionCustom);
}
