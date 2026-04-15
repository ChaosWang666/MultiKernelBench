
#include "rms_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    RmsNormCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int32_t dimCount = xShape->GetStorageShape().GetDimNum();
    
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t numFeatures = xShape->GetStorageShape().GetDim(1);
    
    uint32_t spatialSize = 1;
    for (int32_t i = 2; i < dimCount; i++) {
        spatialSize *= xShape->GetStorageShape().GetDim(i);
    }
    
    const auto* attrs = context->GetAttrs();
    float eps = *(attrs->GetAttrPointer<float>(1));
    
    tiling.set_batchSize(batchSize);
    tiling.set_numFeatures(numFeatures);
    tiling.set_spatialSize(spatialSize);
    tiling.set_eps(eps);
    
    context->SetBlockDim(BLOCK_DIM);
    
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
class RmsNormCustom : public OpDef {
public:
    explicit RmsNormCustom(const char* name) : OpDef(name)
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
        this->Attr("num_features")
            .Int();
        this->Attr("eps")
            .Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(RmsNormCustom);
}
