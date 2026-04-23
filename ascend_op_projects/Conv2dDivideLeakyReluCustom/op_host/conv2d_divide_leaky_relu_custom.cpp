
#include "conv2d_divide_leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t BUFFER_NUM_H = 2;
const uint32_t MAX_TILE_LENGTH = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dDivideLeakyReluCustomTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    uint32_t blockDim = BLOCK_DIM;
    while (blockDim > 1 && totalLength % blockDim != 0) {
        blockDim--;
    }
    context->SetBlockDim(blockDim);

    uint32_t blockLength = totalLength / blockDim;
    uint32_t minTileNum = (blockLength + MAX_TILE_LENGTH * BUFFER_NUM_H - 1) / (MAX_TILE_LENGTH * BUFFER_NUM_H);
    if (minTileNum == 0) {
        minTileNum = 1;
    }

    uint32_t tileNum = minTileNum;
    while (tileNum * BUFFER_NUM_H <= blockLength && blockLength % (tileNum * BUFFER_NUM_H) != 0) {
        tileNum++;
    }
    if (tileNum * BUFFER_NUM_H > blockLength) {
        tileNum = 1;
    }

    auto attrs = context->GetAttrs();
    const float* divisorPtr = attrs->GetAttrPointer<float>(0);
    float divisor = (divisorPtr != nullptr) ? *divisorPtr : 1.0f;

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_divisor(divisor);
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
class Conv2dDivideLeakyReluCustom : public OpDef {
public:
    explicit Conv2dDivideLeakyReluCustom(const char* name) : OpDef(name)
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
        this->Attr("divisor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dDivideLeakyReluCustom);
}
