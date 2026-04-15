
#include "matmul_max_pool_sum_scale_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulMaxPoolSumScaleCustomTilingData tiling;
    
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t outFeatures = x_shape->GetDim(1);
    
    // These are passed via attrs
    const uint32_t kernelSize = 2;
    const float scaleFactor = 0.5f;
    uint32_t pooledLen = outFeatures / kernelSize;
    
    uint32_t BLOCK_DIM = batchSize < 32 ? batchSize : 32;
    uint32_t tileNum = 8;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_outFeatures(outFeatures);
    tiling.set_kernelSize(kernelSize);
    tiling.set_scaleFactor(scaleFactor);
    tiling.set_pooledLen(pooledLen);
    tiling.set_tileNum(tileNum);
    
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
    // Output shape is (batchSize,)
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
class MatmulMaxPoolSumScaleCustom : public OpDef {
public:
    explicit MatmulMaxPoolSumScaleCustom(const char* name) : OpDef(name)
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

OP_ADD(MatmulMaxPoolSumScaleCustom);
}
