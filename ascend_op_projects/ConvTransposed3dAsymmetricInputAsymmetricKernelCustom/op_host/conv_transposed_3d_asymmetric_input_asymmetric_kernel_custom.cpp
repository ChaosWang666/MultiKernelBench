
#include "conv_transposed_3d_asymmetric_input_asymmetric_kernel_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dAsymmetricInputAsymmetricKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* outputShape = context->GetOutputShape(0);

    tiling.set_batch(outputShape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(inputShape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(outputShape->GetOriginShape().GetDim(1));
    tiling.set_depthIn(inputShape->GetOriginShape().GetDim(2));
    tiling.set_heightIn(inputShape->GetOriginShape().GetDim(3));
    tiling.set_widthIn(inputShape->GetOriginShape().GetDim(4));
    tiling.set_depthOut(outputShape->GetOriginShape().GetDim(2));
    tiling.set_heightOut(outputShape->GetOriginShape().GetDim(3));
    tiling.set_widthOut(outputShape->GetOriginShape().GetDim(4));

    tiling.set_kernelDepth(weightShape->GetOriginShape().GetDim(2));
    tiling.set_kernelHeight(weightShape->GetOriginShape().GetDim(3));
    tiling.set_kernelWidth(weightShape->GetOriginShape().GetDim(4));

    tiling.set_strideDepth(1); // Assuming default stride for now
    tiling.set_strideHeight(1);
    tiling.set_strideWidth(1);

    tiling.set_padDepth(0); // Assuming default padding for now
    tiling.set_padHeight(0);
    tiling.set_padWidth(0);

    tiling.set_outPadDepth(0); // Assuming default output padding for now
    tiling.set_outPadHeight(0);
    tiling.set_outPadWidth(0);

    tiling.set_groups(1); // Assuming default groups for now

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
    
    // Placeholder logic - actual implementation would compute output dimensions
    outputShape->SetDim(0, inputShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(1, weightShape->GetOriginShape().GetDim(0));
    outputShape->SetDim(2, 1); // Placeholder
    outputShape->SetDim(3, 1); // Placeholder
    outputShape->SetDim(4, 1); // Placeholder
    
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
class ConvTransposed3dAsymmetricInputAsymmetricKernelCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputAsymmetricKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed3dAsymmetricInputAsymmetricKernelCustom);
}
