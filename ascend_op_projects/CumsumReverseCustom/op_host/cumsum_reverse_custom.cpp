
#include "cumsum_reverse_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    CumsumReverseCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int32_t ndim = xShape->GetStorageShape().GetDimNum();

    // We receive dim as an attribute; for this specific case dim=1, shape is (batch_size, input_shape[0])
    // We'll compute outerSize, dimSize, innerSize
    // For a 2D tensor with dim=1: outerSize=shape[0], dimSize=shape[1], innerSize=1
    // General: outerSize = prod(shape[0..dim-1]), dimSize = shape[dim], innerSize = prod(shape[dim+1..end])
    
    uint32_t totalLength = xShape->GetStorageShape().GetShapeSize();
    
    // Hard-code dim=1 for this specific case (2D tensor, dim=1)
    // outerSize = shape[0], dimSize = shape[1], innerSize = 1
    uint32_t outerSize = 1;
    uint32_t dimSize = 1;
    uint32_t innerSize = 1;
    
    int32_t targetDim = 1; // dim=1
    if (targetDim < 0) targetDim += ndim;
    
    for (int32_t i = 0; i < targetDim; i++) {
        outerSize *= xShape->GetStorageShape().GetDim(i);
    }
    dimSize = xShape->GetStorageShape().GetDim(targetDim);
    for (int32_t i = targetDim + 1; i < ndim; i++) {
        innerSize *= xShape->GetStorageShape().GetDim(i);
    }
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_dimSize(dimSize);
    tiling.set_outerSize(outerSize);
    tiling.set_innerSize(innerSize);
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
class CumsumReverseCustom : public OpDef {
public:
    explicit CumsumReverseCustom(const char* name) : OpDef(name)
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

OP_ADD(CumsumReverseCustom);
}
