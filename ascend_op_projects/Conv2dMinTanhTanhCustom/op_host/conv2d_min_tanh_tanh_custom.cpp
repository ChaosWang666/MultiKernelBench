
#include "conv2d_min_tanh_tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dMinTanhTanhCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = shape.GetDim(0);
    uint32_t C = shape.GetDim(1);
    uint32_t H = shape.GetDim(2);
    uint32_t W = shape.GetDim(3);
    uint32_t HW = H * W;

    uint32_t blockDim = 32;
    if (N < blockDim) {
        blockDim = N;
    }
    uint32_t batchPerBlock = (N + blockDim - 1) / blockDim;

    context->SetBlockDim(blockDim);
    tiling.set_batchPerBlock(batchPerBlock);
    tiling.set_channels(C);
    tiling.set_hw(HW);
    tiling.set_totalBatch(N);
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
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, x_shape->GetDim(2));
    y_shape->SetDim(3, x_shape->GetDim(3));
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
class Conv2dMinTanhTanhCustom : public OpDef {
public:
    explicit Conv2dMinTanhTanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(Conv2dMinTanhTanhCustom);
}
