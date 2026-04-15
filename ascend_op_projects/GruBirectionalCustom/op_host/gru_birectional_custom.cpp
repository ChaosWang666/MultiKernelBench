
#include "gru_bidirectional_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_SEQ_LEN = 64;
const uint32_t TILE_BATCH_SIZE = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GruBirectionalCustomTilingData tiling;
    uint32_t seqLen = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t batchSize = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t inputSize = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    uint32_t hiddenSize = context->GetInputShape(1)->GetOriginShape().GetDim(2);
    uint32_t numLayers = context->GetInputShape(1)->GetOriginShape().GetDim(0) / 2;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_seqLen(seqLen);
    tiling.set_batchSize(batchSize);
    tiling.set_inputSize(inputSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numLayers(numLayers);
    tiling.set_tileSeqLen(TILE_SEQ_LEN);
    tiling.set_tileBatchSize(TILE_BATCH_SIZE);
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
    const gert::Shape* hx_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape* hy_shape = context->GetOutputShape(1);
    
    *y_shape = *x_shape;
    y_shape->GetOriginShape().SetDim(2, hx_shape->GetOriginShape().GetDim(2) * 2); // bidirectional
    
    *hy_shape = *hx_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GruBirectionalCustom : public OpDef {
public:
    explicit GruBirectionalCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("hx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w_hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("hy")
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

OP_ADD(GruBirectionalCustom);
}
