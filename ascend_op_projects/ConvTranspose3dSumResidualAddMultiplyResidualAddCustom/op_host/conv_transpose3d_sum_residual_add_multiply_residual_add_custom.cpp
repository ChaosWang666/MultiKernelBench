
#include "conv_transpose3d_sum_residual_add_multiply_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dSumResidualAddMultiplyResidualAddCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(shape.GetDim(1));
    uint32_t D = static_cast<uint32_t>(shape.GetDim(2));
    uint32_t H = static_cast<uint32_t>(shape.GetDim(3));
    uint32_t W = static_cast<uint32_t>(shape.GetDim(4));

    uint32_t totalFeatureMaps = N * C;
    uint32_t featureMapSize = D * H * W;
    uint32_t channels = C;
    uint32_t tileLength = 8192;
    if (tileLength > featureMapSize) {
        tileLength = featureMapSize;
    }

    uint32_t blockDim = 40;
    if (totalFeatureMaps < blockDim) {
        blockDim = totalFeatureMaps;
    }
    if (blockDim == 0) blockDim = 1;

    context->SetBlockDim(blockDim);
    tiling.set_totalFeatureMaps(totalFeatureMaps);
    tiling.set_featureMapSize(featureMapSize);
    tiling.set_channels(channels);
    tiling.set_tileLength(tileLength);
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
class ConvTranspose3dSumResidualAddMultiplyResidualAddCustom : public OpDef {
public:
    explicit ConvTranspose3dSumResidualAddMultiplyResidualAddCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose3dSumResidualAddMultiplyResidualAddCustom);
}
