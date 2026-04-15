
#include "gemm_batch_norm_scaling_softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmBatchNormScalingSoftmaxCustomTilingData tiling;
    uint32_t batchSize = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t inFeatures = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t outFeatures = context->GetOutputShape(0)->GetOriginShape().GetDim(1);
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inFeatures(inFeatures);
    tiling.set_outFeatures(outFeatures);
    tiling.set_bnEps(1e-5f);
    tiling.set_bnMomentum(0.1f);
    tiling.set_scaleShape0(1);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
class GemmBatchNormScalingSoftmaxCustom : public OpDef {
public:
    explicit GemmBatchNormScalingSoftmaxCustom(const char* name) : OpDef(name)
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

OP_ADD(GemmBatchNormScalingSoftmaxCustom);
}
