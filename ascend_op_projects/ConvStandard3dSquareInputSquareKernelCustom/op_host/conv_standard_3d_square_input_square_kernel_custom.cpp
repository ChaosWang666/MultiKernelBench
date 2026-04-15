
#include "conv_standard_3d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvStandard3dSquareInputSquareKernelCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    tiling.set_batchSize(input_dims[0]);
    tiling.set_inChannels(input_dims[1]);
    tiling.set_outChannels(weight_dims[0]);
    tiling.set_depth(input_dims[2]);
    tiling.set_width(input_dims[3]);
    tiling.set_height(input_dims[4]);
    tiling.set_kernelSize(weight_dims[2]);
    tiling.set_stride(1); // Assuming stride 1 for simplicity
    tiling.set_padding(0); // Assuming padding 0 for simplicity
    tiling.set_dilation(1); // Assuming dilation 1 for simplicity
    tiling.set_groups(1); // Assuming groups 1 for simplicity

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
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* output_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    std::vector<int64_t> output_dims = {input_dims[0], weight_dims[0], input_dims[2], input_dims[3], input_dims[4]};
    *output_shape = gert::Shape(output_dims);
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
class ConvStandard3dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvStandard3dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIJK})
            .UnknownShapeFormat({ge::FORMAT_OIJK});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvStandard3dSquareInputSquareKernelCustom);
}
