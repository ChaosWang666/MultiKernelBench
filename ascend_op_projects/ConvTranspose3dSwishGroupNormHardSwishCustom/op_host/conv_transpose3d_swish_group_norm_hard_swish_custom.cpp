
#include "conv_transpose3d_swish_group_norm_hard_swish_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dSwishGroupNormHardSwishCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = input_shape->GetOriginShape().GetDims();
    tiling.set_batch(shape[0]);
    tiling.set_inChannels(shape[1]);
    tiling.set_outChannels(shape[1]); // Assuming out channels same as in channels for simplicity
    tiling.set_depth(shape[2]);
    tiling.set_height(shape[3]);
    tiling.set_width(shape[4]);
    tiling.set_kernelSize(context->GetAttrInt("kernel_size"));
    tiling.set_stride(context->GetAttrInt("stride"));
    tiling.set_padding(context->GetAttrInt("padding"));
    tiling.set_groups(context->GetAttrInt("groups"));
    tiling.set_eps(context->GetAttrFloat("eps"));

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
class ConvTranspose3dSwishGroupNormHardSwishCustom : public OpDef {
public:
    explicit ConvTranspose3dSwishGroupNormHardSwishCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose3dSwishGroupNormHardSwishCustom);
}
