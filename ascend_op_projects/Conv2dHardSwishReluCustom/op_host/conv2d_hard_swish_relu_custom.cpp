
#include "conv2d_hard_swish_relu_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 20;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dHardSwishReluCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    auto wShape = context->GetInputShape(1)->GetOriginShape();

    uint32_t batchSize = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t inChannels = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t height = static_cast<uint32_t>(xShape.GetDim(2));
    uint32_t width = static_cast<uint32_t>(xShape.GetDim(3));
    uint32_t outChannels = static_cast<uint32_t>(wShape.GetDim(0));
    uint32_t kernelSize = static_cast<uint32_t>(wShape.GetDim(2));
    uint32_t outHeight = height - kernelSize + 1;
    uint32_t totalTasks = batchSize * outHeight;
    uint32_t tasksPerCore = (totalTasks + BLOCK_DIM - 1) / BLOCK_DIM;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inChannels(inChannels);
    tiling.set_outChannels(outChannels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_kernelSize(kernelSize);
    tiling.set_totalTasks(totalTasks);
    tiling.set_tasksPerCore(tasksPerCore);

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
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* wShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);

    int64_t batchSize = xShape->GetDim(0);
    int64_t height = xShape->GetDim(2);
    int64_t width = xShape->GetDim(3);
    int64_t outChannels = wShape->GetDim(0);
    int64_t kernelSize = wShape->GetDim(2);

    yShape->SetDimNum(4);
    yShape->SetDim(0, batchSize);
    yShape->SetDim(1, outChannels);
    yShape->SetDim(2, height - kernelSize + 1);
    yShape->SetDim(3, width - kernelSize + 1);

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
class Conv2dHardSwishReluCustom : public OpDef {
public:
    explicit Conv2dHardSwishReluCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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

OP_ADD(Conv2dHardSwishReluCustom);
}
