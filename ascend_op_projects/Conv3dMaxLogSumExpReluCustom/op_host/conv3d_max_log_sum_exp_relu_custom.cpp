
#include "conv3d_max_log_sum_exp_relu_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMaxLogSumExpReluCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetShapeVector();
    tiling.set_batchSize(shape[0]);
    tiling.set_inChannels(shape[1]);
    tiling.set_depth(shape[2]);
    tiling.set_height(shape[3]);
    tiling.set_width(shape[4]);

    // Set other parameters from attributes
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));

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
    gert::Shape* outputShape = context->GetOutputShape(0);
    const std::vector<int64_t>& inputVec = inputShape->GetOriginShape().GetShapeVector();
    std::vector<int64_t> outputVec = {inputVec[0], inputVec[1], inputVec[2], inputVec[3], inputVec[4]};
    *outputShape = gert::Shape(outputVec);
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
class Conv3dMaxLogSumExpReluCustom : public OpDef {
public:
    explicit Conv3dMaxLogSumExpReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});

        this->Attr("in_channels").SetType(ATTR_TYPE_INT).SetParamType(REQUIRED);
        this->Attr("out_channels").SetType(ATTR_TYPE_INT).SetParamType(REQUIRED);
        this->Attr("kernel_size").SetType(ATTR_TYPE_INT).SetParamType(REQUIRED);
        this->Attr("stride").SetType(ATTR_TYPE_INT).SetParamType(REQUIRED);
        this->Attr("padding").SetType(ATTR_TYPE_INT).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv3dMaxLogSumExpReluCustom);
}
