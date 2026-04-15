
#include "conv3d_multiply_instance_norm_clamp_multiply_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData tiling;
    const gert::Shape* input_shape = context->GetInputShape(0);
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    const gert::Shape* multiplier_shape = context->GetInputShape(3);
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();

    tiling.set_batchSize(input_dims[0]);
    tiling.set_inChannels(input_dims[1]);
    tiling.set_outChannels(weight_dims[0]);
    tiling.set_depth(input_dims[2]);
    tiling.set_height(input_dims[3]);
    tiling.set_width(input_dims[4]);
    tiling.set_kernelDepth(weight_dims[2]);
    tiling.set_kernelHeight(weight_dims[3]);
    tiling.set_kernelWidth(weight_dims[4]);
    tiling.set_padDepth(0);
    tiling.set_padHeight(0);
    tiling.set_padWidth(0);
    tiling.set_clampMin(-1.0f);
    tiling.set_clampMax(1.0f);

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
    const std::vector<int64_t>& input_dims = input_shape->GetOriginShape().GetDims();
    const std::vector<int64_t>& weight_dims = weight_shape->GetOriginShape().GetDims();
    gert::Shape* output_shape = context->GetOutputShape(0);
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
class Conv3dMultiplyInstanceNormClampMultiplyMaxCustom : public OpDef {
public:
    explicit Conv3dMultiplyInstanceNormClampMultiplyMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_OIJKL})
            .UnknownShapeFormat({ge::FORMAT_OIJKL});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_N})
            .UnknownShapeFormat({ge::FORMAT_N});
        this->Input("multiplier")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW});
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

OP_ADD(Conv3dMultiplyInstanceNormClampMultiplyMaxCustom);
}
