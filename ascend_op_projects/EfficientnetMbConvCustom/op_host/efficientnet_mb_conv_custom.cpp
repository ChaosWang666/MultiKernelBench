
#include "efficientnet_mb_conv_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    EfficientnetMbConvCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(input_dims[0]);
    uint32_t inChannels = static_cast<uint32_t>(input_dims[1]);
    uint32_t height = static_cast<uint32_t>(input_dims[2]);
    uint32_t width = static_cast<uint32_t>(input_dims[3]);
    
    uint32_t outChannels = static_cast<uint32_t>(context->GetOutputShape(0)->GetOriginShape().GetDims()[1]);
    uint32_t kernelSize = 5;
    uint32_t stride = 2;
    uint32_t expandRatio = 6;
    uint32_t useResidual = 0;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_stride(stride);
    tiling.set_expandRatio(expandRatio);
    tiling.set_useResidual(useResidual);
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
    const gert::Shape* input_shape = context->GetInputShape(0);
    gert::Shape* output_shape = context->GetOutputShape(0);
    *output_shape = *input_shape;
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
class EfficientnetMbConvCustom : public OpDef {
public:
    explicit EfficientnetMbConvCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("expand_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("depthwise_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("project_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});
        this->Input("expand_bn_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Input("expand_bn_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Input("depthwise_bn_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Input("depthwise_bn_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Input("project_bn_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Input("project_bn_bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(EfficientnetMbConvCustom);
}
