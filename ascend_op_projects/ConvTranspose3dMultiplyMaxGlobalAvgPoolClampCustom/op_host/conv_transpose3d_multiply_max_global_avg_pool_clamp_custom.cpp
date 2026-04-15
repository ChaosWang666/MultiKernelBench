
#include "conv_transpose3d_multiply_max_global_avg_pool_clamp_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustomTilingData tiling;
    const gert::Shape* inputShape = context->GetInputShape(0);
    const std::vector<int64_t>& shape = inputShape->GetOriginShape().GetDims();
    tiling.set_batchSize(shape[0]);
    tiling.set_inChannels(shape[1]);
    tiling.set_outChannels(shape[2]);
    tiling.set_depth(shape[3]);
    tiling.set_height(shape[4]);
    tiling.set_width(shape[5]);
    tiling.set_kernelSize(3);
    tiling.set_stride(2);
    tiling.set_padding(1);
    tiling.set_scale(0.5f);
    tiling.set_maxpoolKernelSize(2);
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
class ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom : public OpDef {
public:
    explicit ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
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

OP_ADD(ConvTranspose3dMultiplyMaxGlobalAvgPoolClampCustom);
}
