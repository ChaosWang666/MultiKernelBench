
#include "conv3d_leaky_relu_sum_clamp_gelu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t ROW_TILE_SIZE = 8192;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dLeakyReluSumClampGeluCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t totalDims = xShape.GetDimNum();

    uint32_t N = (totalDims > 0) ? static_cast<uint32_t>(xShape.GetDim(0)) : 1;
    uint32_t C = (totalDims > 1) ? static_cast<uint32_t>(xShape.GetDim(1)) : 1;
    uint32_t spatialSize = 1;
    for (uint32_t i = 2; i < totalDims; i++) {
        spatialSize *= static_cast<uint32_t>(xShape.GetDim(i));
    }

    uint32_t totalRows = N * C;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalRows(totalRows);
    tiling.set_channelSize(C);
    tiling.set_spatialSize(spatialSize);
    tiling.set_rowTileSize(ROW_TILE_SIZE);

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
class Conv3dLeakyReluSumClampGeluCustom : public OpDef {
public:
    explicit Conv3dLeakyReluSumClampGeluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sum_tensor")
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

OP_ADD(Conv3dLeakyReluSumClampGeluCustom);
}
