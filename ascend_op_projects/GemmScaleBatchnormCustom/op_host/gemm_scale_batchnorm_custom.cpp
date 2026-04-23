
#include "gemm_scale_batchnorm_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmScaleBatchnormCustomTilingData tiling;
    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t batchSize = xShape.GetDim(0);
    uint32_t features = xShape.GetDim(1);

    uint32_t blockDim = BLOCK_DIM;
    if (batchSize < blockDim) {
        blockDim = batchSize;
    }
    uint32_t rowsPerCore = (batchSize + blockDim - 1) / blockDim;
    uint32_t tileRows = rowsPerCore;

    context->SetBlockDim(blockDim);
    tiling.set_batchSize(batchSize);
    tiling.set_features(features);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_tileRows(tileRows);

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
    *y_shape = *x_shape;
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
class GemmScaleBatchnormCustom : public OpDef {
public:
    explicit GemmScaleBatchnormCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("b")
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

OP_ADD(GemmScaleBatchnormCustom);
}
