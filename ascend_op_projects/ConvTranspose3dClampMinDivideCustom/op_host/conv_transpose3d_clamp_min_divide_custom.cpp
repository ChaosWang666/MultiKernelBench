
#include "conv_transpose3d_clamp_min_divide_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dClampMinDivideCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetShapeVector();
    tiling.set_batchSize(shape[0]);
    tiling.set_inChannels(shape[1]);
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));
    tiling.set_depth(shape[2]);
    tiling.set_height(shape[3]);
    tiling.set_width(shape[4]);
    tiling.set_minValue(context->GetAttrFloat("min_value"));
    tiling.set_divisor(context->GetAttrFloat("divisor"));

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
    const std::vector<int64_t>& inputVec = inputShape->GetOriginShape().GetShapeVector();
    gert::Shape* outputShape = context->GetOutputShape(0);
    int64_t batch = inputVec[0];
    int64_t outChannels = context->GetAttrInt("out_channels");
    int64_t kernelSize = context->GetAttrInt("kernel_size");
    int64_t stride = context->GetAttrInt("stride");
    int64_t padding = context->GetAttrInt("padding");
    int64_t depth = inputVec[2];
    int64_t height = inputVec[3];
    int64_t width = inputVec[4];

    int64_t outDepth = (depth - 1) * stride - 2 * padding + kernelSize;
    int64_t outHeight = (height - 1) * stride - 2 * padding + kernelSize;
    int64_t outWidth = (width - 1) * stride - 2 * padding + kernelSize;

    *outputShape = gert::Shape({batch, outChannels, outDepth, outHeight, outWidth});
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
class ConvTranspose3dClampMinDivideCustom : public OpDef {
public:
    explicit ConvTranspose3dClampMinDivideCustom(const char* name) : OpDef(name)
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
        this->Attr("min_value").SetType(ATTR_TYPE_FLOAT).SetParamType(REQUIRED);
        this->Attr("divisor").SetType(ATTR_TYPE_FLOAT).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dClampMinDivideCustom);
}
