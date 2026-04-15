
#include "conv2d_gelu_global_avg_pool_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 1024;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dGeluGlobalAvgPoolCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    uint32_t batchSize = static_cast<uint32_t>(input_dims[0]);
    uint32_t inChannels = static_cast<uint32_t>(input_dims[1]);
    uint32_t outChannels = static_cast<uint32_t>(weight_dims[0]);
    uint32_t height = static_cast<uint32_t>(input_dims[2]);
    uint32_t width = static_cast<uint32_t>(input_dims[3]);
    uint32_t kernelH = static_cast<uint32_t>(weight_dims[2]);
    uint32_t kernelW = static_cast<uint32_t>(weight_dims[3]);
    uint32_t padH = 1;
    uint32_t padW = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelH(kernelH);
    tiling.set_kernelW(kernelW);
    tiling.set_padH(padH);
    tiling.set_padW(padW);
    tiling.set_strideH(strideH);
    tiling.set_strideW(strideW);
    tiling.set_dilationH(dilationH);
    tiling.set_dilationW(dilationW);
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetOriginShape(gert::Shape({input_dims[0], weight_dims[0]}));
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
class Conv2dGeluGlobalAvgPoolCustom : public OpDef {
public:
    explicit Conv2dGeluGlobalAvgPoolCustom(const char* name) : OpDef(name)
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
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_N})
            .UnknownShapeFormat({ge::FORMAT_N});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NC})
            .UnknownShapeFormat({ge::FORMAT_NC});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dGeluGlobalAvgPoolCustom);
}
