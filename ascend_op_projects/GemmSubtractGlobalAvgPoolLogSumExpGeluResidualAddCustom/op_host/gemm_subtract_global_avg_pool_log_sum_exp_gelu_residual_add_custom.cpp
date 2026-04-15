
#include "gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData tiling;
    
    const gert::Shape* gemm_shape = context->GetInputShape(0);
    const gert::Shape* orig_shape = context->GetInputShape(2);
    
    uint32_t batchSize = gemm_shape->GetDim(0);
    uint32_t outFeatures = gemm_shape->GetDim(1);
    uint32_t inFeatures = orig_shape->GetDim(1);
    
    tiling.set_batchSize(batchSize);
    tiling.set_outFeatures(outFeatures);
    tiling.set_inFeatures(inFeatures);
    
    uint32_t blockDim = batchSize < 32 ? batchSize : 32;
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
    const gert::Shape* orig_shape = context->GetInputShape(2);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *orig_shape;
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
class GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom : public OpDef {
public:
    explicit GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom(const char* name) : OpDef(name)
    {
        this->Input("gemm_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("subtract_vec")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("original_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
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

OP_ADD(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom);
}
