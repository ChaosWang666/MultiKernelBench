
#include "conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 128;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t channels = xShape->GetStorageShape().GetDim(1);
    uint32_t dim2 = xShape->GetStorageShape().GetDim(2);
    uint32_t dim3 = xShape->GetStorageShape().GetDim(3);
    uint32_t dim4 = xShape->GetStorageShape().GetDim(4);
    uint32_t spatialSize = dim2 * dim3 * dim4;
    uint32_t totalOutput = batchSize * spatialSize;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_spatialSize(spatialSize);
    tiling.set_totalOutput(totalOutput);
    tiling.set_tileNum(TILE_NUM);
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
    // Output shape: (batch, 1, D, H, W) -- but we flatten to (totalOutput,)
    // Actually let's set proper output shape
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, x_shape->GetDim(2));
    y_shape->SetDim(3, x_shape->GetDim(3));
    y_shape->SetDim(4, x_shape->GetDim(4));
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
class ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(ConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom);
}
