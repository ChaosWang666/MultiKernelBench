
#include "gemm_group_norm_hardtanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmGroupNormHardtanhCustomTilingData tiling;

    auto xShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t totalRows = (uint32_t)xShape.GetDim(0);
    uint32_t outFeatures = (uint32_t)xShape.GetDim(1);

    auto attrs = context->GetAttrs();
    const int64_t* numGroupsPtr = attrs->GetAttrPointer<int64_t>(0);
    const float* htMinPtr = attrs->GetAttrPointer<float>(1);
    const float* htMaxPtr = attrs->GetAttrPointer<float>(2);
    const float* epsPtr = attrs->GetAttrPointer<float>(3);

    uint32_t numGroups = (uint32_t)(*numGroupsPtr);
    uint32_t blockDim = BLOCK_DIM;
    uint32_t rowsPerBlock = (totalRows + blockDim - 1) / blockDim;

    context->SetBlockDim(blockDim);
    tiling.set_totalRows(totalRows);
    tiling.set_outFeatures(outFeatures);
    tiling.set_numGroups(numGroups);
    tiling.set_rowsPerBlock(rowsPerBlock);
    tiling.set_htMin(*htMinPtr);
    tiling.set_htMax(*htMaxPtr);
    tiling.set_eps(*epsPtr);

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
class GemmGroupNormHardtanhCustom : public OpDef {
public:
    explicit GemmGroupNormHardtanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("num_groups").Int();
        this->Attr("hardtanh_min").Float();
        this->Attr("hardtanh_max").Float();
        this->Attr("eps").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmGroupNormHardtanhCustom);
}
