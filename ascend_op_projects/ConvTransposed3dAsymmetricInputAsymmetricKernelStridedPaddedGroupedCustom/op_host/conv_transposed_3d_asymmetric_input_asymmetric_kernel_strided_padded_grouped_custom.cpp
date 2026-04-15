
#include "conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    tiling.set_batch(inputDims[0]);
    tiling.set_inChannels(inputDims[1]);
    tiling.set_outChannels(weightDims[0]);
    tiling.set_kernelDepth(weightDims[2]);
    tiling.set_kernelHeight(weightDims[3]);
    tiling.set_kernelWidth(weightDims[4]);
    tiling.set_strideDepth(1); // Placeholder, actual value will be passed via parameters
    tiling.set_strideHeight(1); // Placeholder
    tiling.set_strideWidth(1); // Placeholder
    tiling.set_padDepth(0); // Placeholder
    tiling.set_padHeight(0); // Placeholder
    tiling.set_padWidth(0); // Placeholder
    tiling.set_outDepth(1); // Placeholder
    tiling.set_outHeight(1); // Placeholder
    tiling.set_outWidth(1); // Placeholder
    tiling.set_groups(1); // Placeholder
    tiling.set_totalElements(inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * inputDims[4]);

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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* w_shape = context->GetInputShape(1);
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
class ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom : public OpDef {
public:
    explicit ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("w")
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

OP_ADD(ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom);
}
