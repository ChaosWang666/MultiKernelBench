
#include "conv_transpose2d_min_sum_gelu_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dMinSumGeluAddCustomTilingData tiling;
    
    const gert::Shape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetDim(0);
    uint32_t channels = xShape->GetDim(1);
    uint32_t height = xShape->GetDim(2);
    uint32_t width = xShape->GetDim(3);
    
    const gert::Shape* biasShape = context->GetInputShape(1);
    uint32_t biasLength = biasShape->GetOriginShape().GetShapeSize();
    
    // Use batch_size as block dim, one block per batch
    uint32_t blockDim = batchSize;
    if (blockDim > 32) blockDim = 32;
    
    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_channels(channels);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_biasLength(biasLength);
    
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
    // Output shape: [batch, 1, 1, width]
    y_shape->SetDimNum(4);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, 1);
    y_shape->SetDim(2, 1);
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
class ConvTranspose2dMinSumGeluAddCustom : public OpDef {
public:
    explicit ConvTranspose2dMinSumGeluAddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
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

OP_ADD(ConvTranspose2dMinSumGeluAddCustom);
}
