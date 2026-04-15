
#include "conv_transposed_2d_square_input_square_kernel_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTransposed2dSquareInputSquareKernelCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const gert::Shape* weightShape = context->GetInputShape(1);
    const std::vector<int64_t>& inputDims = inputShape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weightShape->GetOriginShape().GetDims();

    tiling.set_batchSize(inputDims[0]);
    tiling.set_inChannels(inputDims[1]);
    tiling.set_outChannels(weightDims[0]);
    tiling.set_height(inputDims[2]);
    tiling.set_width(inputDims[3]);
    tiling.set_kernelSize(weightDims[2]);
    tiling.set_stride(1); // Assuming stride 1 for simplicity
    tiling.set_padding(0); // Assuming padding 0 for simplicity
    tiling.set_outputPadding(0); // Assuming output_padding 0 for simplicity
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
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const std::vector<int64_t>& inputDims = x1_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weightDims = weight_shape->GetOriginShape().GetDims();
    int64_t batch = inputDims[0];
    int64_t outChannels = weightDims[0];
    int64_t height = inputDims[2];
    int64_t width = inputDims[3];
    y_shape->SetDim(0, batch);
    y_shape->SetDim(1, outChannels);
    y_shape->SetDim(2, height);
    y_shape->SetDim(3, width);
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
class ConvTransposed2dSquareInputSquareKernelCustom : public OpDef {
public:
    explicit ConvTransposed2dSquareInputSquareKernelCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTransposed2dSquareInputSquareKernelCustom);
}
