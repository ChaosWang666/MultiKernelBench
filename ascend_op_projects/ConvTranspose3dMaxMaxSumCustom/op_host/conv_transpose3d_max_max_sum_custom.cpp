
#include "conv_transpose3d_max_max_sum_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dMaxMaxSumCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t B = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(shape.GetDim(1));
    uint32_t D = static_cast<uint32_t>(shape.GetDim(2));
    uint32_t H = static_cast<uint32_t>(shape.GetDim(3));
    uint32_t W = static_cast<uint32_t>(shape.GetDim(4));
    uint32_t DHW = D * H * W;

    uint32_t tileLen = 128;
    if (DHW == 0) DHW = 1;
    uint32_t tilesPerBatch = (DHW + tileLen - 1) / tileLen;
    uint32_t totalTiles = B * tilesPerBatch;
    if (totalTiles == 0) totalTiles = 1;

    uint32_t numCores = 20;
    uint32_t tilesPerCore = (totalTiles + numCores - 1) / numCores;
    if (tilesPerCore == 0) tilesPerCore = 1;
    uint32_t usedCores = (totalTiles + tilesPerCore - 1) / tilesPerCore;
    if (usedCores == 0) usedCores = 1;

    context->SetBlockDim(usedCores);
    tiling.set_B(B);
    tiling.set_C(C);
    tiling.set_DHW(DHW);
    tiling.set_tileLen(tileLen);
    tiling.set_tilesPerBatch(tilesPerBatch);
    tiling.set_totalTiles(totalTiles);
    tiling.set_tilesPerCore(tilesPerCore);
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
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    yShape->SetDimNum(5);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, 1);
    yShape->SetDim(2, xShape->GetDim(2));
    yShape->SetDim(3, xShape->GetDim(3));
    yShape->SetDim(4, xShape->GetDim(4));
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
class ConvTranspose3dMaxMaxSumCustom : public OpDef {
public:
    explicit ConvTranspose3dMaxMaxSumCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose3dMaxMaxSumCustom);
}
