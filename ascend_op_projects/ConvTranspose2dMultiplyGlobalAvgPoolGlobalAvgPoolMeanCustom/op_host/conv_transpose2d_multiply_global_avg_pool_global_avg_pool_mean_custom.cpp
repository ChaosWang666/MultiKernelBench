
#include "conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const gert::Shape* outputShape = context->GetOutputShape(0);
    
    tiling.set_batchSize(inputShape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(inputShape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(weightShape->GetOriginShape().GetDim(0));
    tiling.set_height(inputShape->GetOriginShape().GetDim(2));
    tiling.set_width(inputShape->GetOriginShape().GetDim(3));
    tiling.set_kernelH(weightShape->GetOriginShape().GetDim(2));
    tiling.set_kernelW(weightShape->GetOriginShape().GetDim(3));
    tiling.set_strideH(2); // Assuming fixed stride for simplicity
    tiling.set_strideW(2);
    tiling.set_padH(1); // Assuming fixed padding for simplicity
    tiling.set_padW(1);
    tiling.set_outputPadH(1); // Assuming fixed output padding for simplicity
    tiling.set_outputPadW(1);
    tiling.set_multiplier(0.5f); // Assuming fixed multiplier for simplicity
    
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
    const gert::Shape* weight_shape = context->GetInputShape(1);
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
class ConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom : public OpDef {
public:
    explicit ConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIHW})
            .UnknownShapeFormat({ge::FORMAT_OIHW});
        this->OptionalInput("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_N})
            .UnknownShapeFormat({ge::FORMAT_N});
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

OP_ADD(ConvTranspose2dMultiplyGlobalAvgPoolGlobalAvgPoolMeanCustom);
}
