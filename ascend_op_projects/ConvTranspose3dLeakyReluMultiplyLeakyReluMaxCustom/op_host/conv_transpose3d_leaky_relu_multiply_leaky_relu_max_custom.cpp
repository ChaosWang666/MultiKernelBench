
#include "conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    // x shape: [N, C, D, H, W]
    uint32_t N = xShape->GetOriginShape().GetDim(0);
    uint32_t C = xShape->GetOriginShape().GetDim(1);
    uint32_t D = xShape->GetOriginShape().GetDim(2);
    uint32_t H = xShape->GetOriginShape().GetDim(3);
    uint32_t W = xShape->GetOriginShape().GetDim(4);
    
    uint32_t Do = D / 2;
    uint32_t Ho = H / 2;
    uint32_t Wo = W / 2;
    uint32_t totalOut = N * C * Do * Ho * Wo;

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(N);
    tiling.set_channels(C);
    tiling.set_depthIn(D);
    tiling.set_heightIn(H);
    tiling.set_widthIn(W);
    tiling.set_depthOut(Do);
    tiling.set_heightOut(Ho);
    tiling.set_widthOut(Wo);
    tiling.set_totalOutputElems(totalOut);
    tiling.set_tileNum(TILE_NUM);
    
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
    // Input: [N, C, D, H, W], Output: [N, C, D/2, H/2, W/2]
    y_shape->SetDimNum(5);
    y_shape->SetDim(0, x_shape->GetDim(0));
    y_shape->SetDim(1, x_shape->GetDim(1));
    y_shape->SetDim(2, x_shape->GetDim(2) / 2);
    y_shape->SetDim(3, x_shape->GetDim(3) / 2);
    y_shape->SetDim(4, x_shape->GetDim(4) / 2);
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
class ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom : public OpDef {
public:
    explicit ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("multiplier")
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

OP_ADD(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom);
}
