
#include "matmul_max_pool_sum_scale_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BATCHES_PER_BLOCK_HOST = 4;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulMaxPoolSumScaleCustomTilingData tiling;
    auto inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t totalBatches = inputShape.GetDim(0);
    uint32_t features = inputShape.GetDim(1);

    uint32_t blockDim = (totalBatches + BATCHES_PER_BLOCK_HOST - 1) / BATCHES_PER_BLOCK_HOST;
    context->SetBlockDim(blockDim);

    tiling.set_totalBatches(totalBatches);
    tiling.set_features(features);
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
