
#include "windowed_causal_attention_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    WindowedCausalAttentionCustomTilingData tiling;
    // Q shape: [B, N, H, d_h]
    const gert::Shape* qShape = context->GetInputShape(0);
    uint32_t B = qShape->GetDim(0);
    uint32_t N = qShape->GetDim(1);
    uint32_t H = qShape->GetDim(2);
    uint32_t d_h = qShape->GetDim(3);
    uint32_t windowSize = 256;

    // Use B*H blocks so each block handles one (batch, head) pair
    uint32_t blockDim = B * H;
    context->SetBlockDim(blockDim);

    tiling.set_batchSize(B);
    tiling.set_seqLen(N);
    tiling.set_numHeads(H);
    tiling.set_headDim(d_h);
    tiling.set_windowSize(windowSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // Workspace: for each block, we need space for scores (windowSize+1 floats aligned)
    // and temporary storage
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = blockDim * (windowSize + 1) * sizeof(float) * 2;
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
class WindowedCausalAttentionCustom : public OpDef {
public:
    explicit WindowedCausalAttentionCustom(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
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

OP_ADD(WindowedCausalAttentionCustom);
}
