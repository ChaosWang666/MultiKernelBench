
#include "conv2d_mish_mish_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t BUFFER_NUM_VAL = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dMishMishCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);

    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t halfBlock = blockLength / BUFFER_NUM_VAL;

    uint32_t tileLength = 2048;
    while (tileLength >= 8) {
        if (tileLength % 8 == 0 && halfBlock % tileLength == 0) break;
        tileLength -= 8;
    }
    if (tileLength < 8) tileLength = 8;
    uint32_t tileNum = halfBlock / tileLength;
    if (tileNum == 0) tileNum = 1;

    tiling.set_totalLength(totalLength);
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
class Conv2dMishMishCustom : public OpDef {
public:
    explicit Conv2dMishMishCustom(const char* name) : OpDef(name)
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

OP_ADD(Conv2dMishMishCustom);
}
