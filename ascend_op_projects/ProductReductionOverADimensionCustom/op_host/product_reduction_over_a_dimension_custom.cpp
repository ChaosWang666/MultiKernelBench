
#include "product_reduction_over_a_dimension_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ProductReductionOverADimensionCustomTilingData tiling;
    
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    auto shape = inputShape->GetOriginShape();
    int32_t dimCount = shape.GetDimNum();
    
    // Get the dim attribute
    const uint64_t* dimPtr = context->GetAttrs()->GetAttrPointer<uint64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    if (dim < 0) dim += dimCount;
    
    uint32_t totalLength = shape.GetShapeSize();
    uint32_t dimSize = shape.GetDim(dim);
    
    uint32_t outerSize = 1;
    for (int i = 0; i < dim; i++) {
        outerSize *= shape.GetDim(i);
    }
    uint32_t innerSize = 1;
    for (int i = dim + 1; i < dimCount; i++) {
        innerSize *= shape.GetDim(i);
    }
    
    uint32_t outputLength = outerSize * innerSize;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dim(dim);
    tiling.set_dimSize(dimSize);
    tiling.set_outerSize(outerSize);
    tiling.set_innerSize(innerSize);
    tiling.set_outputLength(outputLength);
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
    
    int32_t dimCount = x_shape->GetDimNum();
    const uint64_t* dimPtr = context->GetAttrs()->GetAttrPointer<uint64_t>(0);
    int32_t dim = (int32_t)(*dimPtr);
    if (dim < 0) dim += dimCount;
    
    // Output shape removes the reduction dimension
    int outIdx = 0;
    for (int i = 0; i < dimCount; i++) {
        if (i != dim) {
            y_shape->SetDim(outIdx++, x_shape->GetDim(i));
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ProductReductionOverADimensionCustom : public OpDef {
public:
    explicit ProductReductionOverADimensionCustom(const char* name) : OpDef(name)
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

OP_ADD(ProductReductionOverADimensionCustom);
}
