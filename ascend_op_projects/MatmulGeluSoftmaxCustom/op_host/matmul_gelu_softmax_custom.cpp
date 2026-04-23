
#include "matmul_gelu_softmax_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 20;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulGeluSoftmaxCustomTilingData tiling;
    const gert::Shape& shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t rows = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t cols = static_cast<uint32_t>(shape.GetDim(1));

    uint32_t blockDim = (rows < BLOCK_DIM) ? rows : BLOCK_DIM;
    if (blockDim == 0) {
        blockDim = 1;
    }
    context->SetBlockDim(blockDim);
    tiling.set_rows(rows);
    tiling.set_cols(cols);
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
class MatmulGeluSoftmaxCustom : public OpDef {
public:
    explicit MatmulGeluSoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(MatmulGeluSoftmaxCustom);
}
