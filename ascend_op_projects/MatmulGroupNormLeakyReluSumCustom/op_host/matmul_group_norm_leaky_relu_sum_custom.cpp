
#include "matmul_group_norm_leaky_relu_sum_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t MAX_BLOCK_DIM = 20;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulGroupNormLeakyReluSumCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = xShape.GetDim(0);
    uint32_t hiddenSize = xShape.GetDim(1);
    uint32_t numGroups = 512;
    if (hiddenSize % numGroups != 0) {
        numGroups = 1;
    }
    uint32_t channelsPerGroup = hiddenSize / numGroups;

    uint32_t blockDim = (batchSize < MAX_BLOCK_DIM) ? batchSize : MAX_BLOCK_DIM;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    tiling.set_batchSize(batchSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numGroups(numGroups);
    tiling.set_channelsPerGroup(channelsPerGroup);

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
class MatmulGroupNormLeakyReluSumCustom : public OpDef {
public:
    explicit MatmulGroupNormLeakyReluSumCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
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

OP_ADD(MatmulGroupNormLeakyReluSumCustom);
}
