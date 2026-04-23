
#include "conv_transpose2d_min_sum_gelu_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConvTranspose2dMinSumGeluAddCustomTilingData tiling;
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t totalN = xShape->GetOriginShape().GetDim(0);
    uint32_t totalC = xShape->GetOriginShape().GetDim(1);
    uint32_t totalH = xShape->GetOriginShape().GetDim(2);
    uint32_t totalW = xShape->GetOriginShape().GetDim(3);

    tiling.set_totalN(totalN);
    tiling.set_totalC(totalC);
    tiling.set_totalH(totalH);
    tiling.set_totalW(totalW);

    uint32_t blockDim = totalN;
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

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
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ConvTranspose2dMinSumGeluAddCustom);
}
