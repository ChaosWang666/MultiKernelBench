
#include "matmul_avg_pool_gelu_scale_max_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulAvgPoolGeluScaleMaxCustomTilingData tiling;
    const gert::Shape* x_shape = context->GetInputShape(0);
    uint32_t batchSize = x_shape->GetDim(0);
    uint32_t inputLength = x_shape->GetDim(1);

    // Get attrs
    const auto* attrs = context->GetAttrs();
    uint32_t poolKernelSize = *(attrs->GetAttrPointer<uint32_t>(0));
    float scaleFactor = *(attrs->GetAttrPointer<float>(1));

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_inputLength(inputLength);
    tiling.set_poolKernelSize(poolKernelSize);
    tiling.set_scaleFactor(scaleFactor);

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
    // Output shape is (batchSize,)
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x_shape->GetDim(0));
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
class MatmulAvgPoolGeluScaleMaxCustom : public OpDef {
public:
    explicit MatmulAvgPoolGeluScaleMaxCustom(const char* name) : OpDef(name)
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
        this->Attr("pool_kernel_size").AttrType(REQUIRED).Int();
        this->Attr("scale_factor").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulAvgPoolGeluScaleMaxCustom);
}
