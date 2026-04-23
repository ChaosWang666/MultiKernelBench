
#include "matmul_avg_pool_gelu_scale_max_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t POOL_KERNEL_SIZE = 16;
const float SCALE_FACTOR = 2.0f;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulAvgPoolGeluScaleMaxCustomTilingData tiling;
    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = inputShape.GetDim(0);
    uint32_t outFeatures = inputShape.GetDim(1);
    uint32_t pooledSize = outFeatures / POOL_KERNEL_SIZE;
    float invPoolSize = 1.0f / (float)POOL_KERNEL_SIZE;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_outFeatures(outFeatures);
    tiling.set_poolKernelSize(POOL_KERNEL_SIZE);
    tiling.set_pooledSize(pooledSize);
    tiling.set_scaleFactor(SCALE_FACTOR);
    tiling.set_invPoolSize(invPoolSize);

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
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x_shape->GetDim(0));
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
class MatmulAvgPoolGeluScaleMaxCustom : public OpDef {
public:
    explicit MatmulAvgPoolGeluScaleMaxCustom(const char* name) : OpDef(name)
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

OP_ADD(MatmulAvgPoolGeluScaleMaxCustom);
}
