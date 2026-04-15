
#include "grid_sample_random_warp_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 4;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GridSampleRandomWarpCustomTilingData tiling;
    // x shape: [N, C, H, W]
    auto x_shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = x_shape.GetDim(0);
    uint32_t C = x_shape.GetDim(1);
    uint32_t inH = x_shape.GetDim(2);
    uint32_t inW = x_shape.GetDim(3);

    // grid shape: [N, outH, outW, 2]
    auto grid_shape = context->GetInputShape(1)->GetOriginShape();
    uint32_t outH = grid_shape.GetDim(1);
    uint32_t outW = grid_shape.GetDim(2);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(N);
    tiling.set_channels(C);
    tiling.set_inH(inH);
    tiling.set_inW(inW);
    tiling.set_outH(outH);
    tiling.set_outW(outW);
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
    const gert::Shape* grid_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    // output: [N, C, outH, outW]
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, grid_shape->GetDim(1));
    y_shape->SetDim(3, grid_shape->GetDim(2));
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
class GridSampleRandomWarpCustom : public OpDef {
public:
    explicit GridSampleRandomWarpCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grid")
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

OP_ADD(GridSampleRandomWarpCustom);
}
