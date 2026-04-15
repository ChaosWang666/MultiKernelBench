
#include "net_vlad_with_ghost_clusters_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 32;
const uint32_t TILE_NUM = 4096;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    NetVladWithGhostClustersCustomTilingData tiling;
    uint32_t batchSize = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    uint32_t numFeatures = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    uint32_t featureSize = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    uint32_t clusterSize = context->GetInputShape(1)->GetOriginShape().GetDim(1);
    uint32_t ghostClusters = context->GetInputShape(1)->GetOriginShape().GetDim(0) - clusterSize;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_batchSize(batchSize);
    tiling.set_numFeatures(numFeatures);
    tiling.set_featureSize(featureSize);
    tiling.set_clusterSize(clusterSize);
    tiling.set_ghostClusters(ghostClusters);
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
    gert::Shape* output_shape = context->GetOutputShape(0);
    uint32_t batchSize = x_shape->GetOriginShape().GetDim(0);
    uint32_t clusterSize = context->GetInputShape(1)->GetOriginShape().GetDim(1);
    uint32_t featureSize = context->GetInputShape(0)->GetOriginShape().GetDim(2);
    output_shape->SetDim(0, batchSize);
    output_shape->SetDim(1, clusterSize * featureSize);
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
class NetVladWithGhostClustersCustom : public OpDef {
public:
    explicit NetVladWithGhostClustersCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clusters")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("clusters2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("batchNormWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("batchNormBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("batchNormMean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("batchNormVar")
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

OP_ADD(NetVladWithGhostClustersCustom);
}
