
#include "conv_transpose3d_scale_batch_norm_global_avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t MAX_BLOCK_DIM = 32;
const uint32_t DEFAULT_TILE_LEN = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t B = (uint32_t)shape.GetDim(0);
    uint32_t C = (uint32_t)shape.GetDim(1);
    uint32_t D = (uint32_t)shape.GetDim(2);
    uint32_t H = (uint32_t)shape.GetDim(3);
    uint32_t W = (uint32_t)shape.GetDim(4);

    uint32_t numGroups = B * C;
    uint32_t groupSize = D * H * W;

    uint32_t numBlocks = (numGroups < MAX_BLOCK_DIM) ? numGroups : MAX_BLOCK_DIM;
    if (numBlocks == 0) numBlocks = 1;
    uint32_t groupsPerBlock = (numGroups + numBlocks - 1) / numBlocks;

    uint32_t tileLength = DEFAULT_TILE_LEN;
    if (groupSize < tileLength) {
        tileLength = ((groupSize + 7) / 8) * 8;
        if (tileLength < 8) tileLength = 8;
    }

    context->SetBlockDim(numBlocks);
    tiling.set_numGroups(numGroups);
    tiling.set_groupSize(groupSize);
    tiling.set_tileLength(tileLength);
    tiling.set_groupsPerBlock(groupsPerBlock);
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
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, 1);
    y_shape->SetDim(3, 1);
    y_shape->SetDim(4, 1);
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
class ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom : public OpDef {
public:
    explicit ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom);
}
