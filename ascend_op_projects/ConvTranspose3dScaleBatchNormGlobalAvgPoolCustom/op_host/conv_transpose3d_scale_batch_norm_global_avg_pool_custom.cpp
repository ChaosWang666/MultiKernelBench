
#include "conv_transpose3d_scale_batch_norm_global_avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData tiling;

    // x shape: [batch, channels, D, H, W]
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t channels = x_shape->GetDim(1);
    uint32_t spatialSize = 1;
    for (int i = 2; i < x_shape->GetDimNum(); i++) {
        spatialSize *= x_shape->GetDim(i);
    }

    // Get attrs from tiling context - we pass scale_factor and eps via tiling
    // For now use defaults that will be overridden
    float scaleFactor = 2.0f;
    float eps = 1e-5f;

    // Try to get attrs
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* sf = attrs->GetAttrPointer<float>(0);
        if (sf) scaleFactor = *sf;
        const float* ep = attrs->GetAttrPointer<float>(1);
        if (ep) eps = *ep;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_spatialSize(spatialSize);
    tiling.set_scaleFactor(scaleFactor);
    tiling.set_eps(eps);
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
    // Output: [batch, channels, 1, 1, 1]
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
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("running_mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("running_var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("scale_factor").AttrType(OPTIONAL).Float(2.0f);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom);
}
