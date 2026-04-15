
#include "conv_transpose2d_gelu_group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dGeluGroupNormCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = input_shape->GetOriginShape().GetDims();
    uint32_t batch = static_cast<uint32_t>(shape[0]);
    uint32_t inChannels = static_cast<uint32_t>(shape[1]);
    uint32_t height = static_cast<uint32_t>(shape[2]);
    uint32_t width = static_cast<uint32_t>(shape[3]);

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch(batch);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(context->GetAttrInt("out_channels"));
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_groups(context->GetAttrInt("groups"));
    tiling.set_numGroups(context->GetAttrInt("num_groups"));
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
class ConvTranspose2dGeluGroupNormCustom : public OpDef {
public:
    explicit ConvTranspose2dGeluGroupNormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});

        this->Attr("in_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("out_channels").SetType(INT).SetParamType(REQUIRED);
        this->Attr("kernel_size").SetType(INT).SetParamType(REQUIRED);
        this->Attr("stride").SetType(INT).SetParamType(REQUIRED);
        this->Attr("groups").SetType(INT).SetParamType(REQUIRED);
        this->Attr("num_groups").SetType(INT).SetParamType(REQUIRED);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dGeluGroupNormCustom);
}
