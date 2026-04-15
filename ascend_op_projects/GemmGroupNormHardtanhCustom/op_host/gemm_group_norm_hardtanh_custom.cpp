
#include "gemm_group_norm_hardtanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmGroupNormHardtanhCustomTilingData tiling;
    uint32_t batch_size = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t in_features = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t out_features = context->GetAttrInt64("out_features");
    uint32_t num_groups = context->GetAttrInt64("num_groups");
    float hardtanh_min = context->GetAttrFloat("hardtanh_min");
    float hardtanh_max = context->GetAttrFloat("hardtanh_max");

    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batch_size(batch_size);
    tiling.set_in_features(in_features);
    tiling.set_out_features(out_features);
    tiling.set_num_groups(num_groups);
    tiling.set_hardtanh_min(hardtanh_min);
    tiling.set_hardtanh_max(hardtanh_max);
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
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("in_features").SetType(ATTR_TYPE_INT64).SetRequired(true);
        this->Attr("out_features").SetType(ATTR_TYPE_INT64).SetRequired(true);
        this->Attr("num_groups").SetType(ATTR_TYPE_INT64).SetRequired(true);
        this->Attr("hardtanh_min").SetType(ATTR_TYPE_FLOAT).SetRequired(true);
        this->Attr("hardtanh_max").SetType(ATTR_TYPE_FLOAT).SetRequired(true);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GemmGroupNormHardtanhCustom);
}
