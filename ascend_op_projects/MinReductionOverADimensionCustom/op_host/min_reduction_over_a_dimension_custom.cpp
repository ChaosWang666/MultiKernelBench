
#include "min_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 16;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MinReductionOverADimensionCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    auto shape = xShape->GetOriginShape();
    uint32_t totalLength = shape.GetShapeSize();
    
    // Get dim attribute
    const uint32_t* dimPtr = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
    uint32_t dim = *dimPtr;
    
    uint32_t ndim = shape.GetDimNum();
    
    // Handle negative dim
    if (dim >= ndim) dim = ndim - 1;
    
    uint32_t dimSize = shape.GetDim(dim);
    
    uint32_t outerSize = 1;
    for (uint32_t i = 0; i < dim; i++) {
        outerSize *= shape.GetDim(i);
    }
    
    uint32_t innerSize = 1;
    for (uint32_t i = dim + 1; i < ndim; i++) {
        innerSize *= shape.GetDim(i);
    }
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dim(dim);
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
    const uint32_t* dimPtr = context->GetAttrs()->GetAttrPointer<uint32_t>(0);
    uint32_t dim = *dimPtr;
    uint32_t ndim = x_shape->GetDimNum();
    
    // Output shape removes the reduced dimension
    uint32_t outIdx = 0;
    for (uint32_t i = 0; i < ndim; i++) {
        if (i != dim) {
            y_shape->SetDim(outIdx, x_shape->GetDim(i));
            outIdx++;
        }
    }
    // Set the number of dims
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
class MinReductionOverADimensionCustom : public OpDef {
public:
    explicit MinReductionOverADimensionCustom(const char* name) : OpDef(name)
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

OP_ADD(MinReductionOverADimensionCustom);
}
