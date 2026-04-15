
#include "convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Convtranspose2dBatchnormTanhMaxpoolGroupnormCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    tiling.set_batchSize(shape[0]);
    tiling.set_inChannels(shape[1]);
    tiling.set_height(shape[2]);
    tiling.set_width(shape[3]);
    tiling.set_outChannels(128); // hardcoded for example
    tiling.set_kernelSize(5); // hardcoded for example
    tiling.set_stride(1); // hardcoded for example
    tiling.set_padding(1); // hardcoded for example
    tiling.set_groups(8); // hardcoded for example
    tiling.set_numGroups(8); // hardcoded for example
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
class Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom : public OpDef {
public:
    explicit Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom(const char* name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Convtranspose2dBatchnormTanhMaxpoolGroupnormCustom);
}
