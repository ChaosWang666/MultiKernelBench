
#include "conv_transpose3d_sum_residual_add_multiply_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dSumResidualAddMultiplyResidualAddCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const gert::Shape* output_shape = context->GetOutputShape(0);

    tiling.set_batch(input_shape->GetOriginShape().GetDim(0));
    tiling.set_inChannels(input_shape->GetOriginShape().GetDim(1));
    tiling.set_outChannels(output_shape->GetOriginShape().GetDim(1));
    tiling.set_depth(output_shape->GetOriginShape().GetDim(2));
    tiling.set_height(output_shape->GetOriginShape().GetDim(3));
    tiling.set_width(output_shape->GetOriginShape().GetDim(4));
    tiling.set_kernelDepth(weight_shape->GetOriginShape().GetDim(2));
    tiling.set_kernelHeight(weight_shape->GetOriginShape().GetDim(3));
    tiling.set_kernelWidth(weight_shape->GetOriginShape().GetDim(4));
    tiling.set_strideDepth(2);
    tiling.set_strideHeight(2);
    tiling.set_strideWidth(2);
    tiling.set_padDepth(1);
    tiling.set_padHeight(1);
    tiling.set_padWidth(1);
    tiling.set_outputPadDepth(1);
    tiling.set_outputPadHeight(1);
    tiling.set_outputPadWidth(1);
    uint32_t totalLength = output_shape->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
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
    const gert::Shape* bias_shape = context->GetInputShape(2);
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
class ConvTranspose3dSumResidualAddMultiplyResidualAddCustom : public OpDef {
public:
    explicit ConvTranspose3dSumResidualAddMultiplyResidualAddCustom(const char* name) : OpDef(name)
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

OP_ADD(ConvTranspose3dSumResidualAddMultiplyResidualAddCustom);
}
