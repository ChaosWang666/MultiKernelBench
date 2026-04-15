
#include "conv2d_batch_norm_scaling_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dBatchNormScalingCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* biasShape = context->GetInputShape(2);
    const gert::Shape* bnWeightShape = context->GetInputShape(3);
    const gert::Shape* bnBiasShape = context->GetInputShape(4);
    const gert::Shape* bnMeanShape = context->GetInputShape(5);
    const gert::Shape* bnVarShape = context->GetInputShape(6);

    tiling.set_batchSize(inputShape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(inputShape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(biasShape->GetOriginShape().GetDim(0));
    tiling.set_height(inputShape->GetOriginShape().GetDim(2));
    tiling.set_width(inputShape->GetOriginShape().GetDim(3));
    tiling.set_kernelH(weightShape->GetOriginShape().GetDim(2));
    tiling.set_kernelW(weightShape->GetOriginShape().GetDim(3));
    tiling.set_padH(0); // Assuming default padding
    tiling.set_padW(0);
    tiling.set_strideH(1); // Assuming default stride
    tiling.set_strideW(1);
    tiling.set_dilationH(1);
    tiling.set_dilationW(1);
    tiling.set_scalingFactor(2.0f); // Assuming fixed scaling factor

    context->SetBlockDim(BLOCK_DIM);
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
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    outputShape->SetDim(0, inputShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(1, weightShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(2, inputShape->GetOriginShape().GetDim(2));
    outputShape->SetDim(3, inputShape->GetOriginShape().GetDim(3));
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
class Conv2dBatchNormScalingCustom : public OpDef {
public:
    explicit Conv2dBatchNormScalingCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("convWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("convBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});
        this->Input("bnWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});
        this->Input("bnBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});
        this->Input("bnMean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});
        this->Input("bnVar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dBatchNormScalingCustom);
}
