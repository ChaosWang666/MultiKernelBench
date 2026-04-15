
#include "group_norm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GroupNormCustomTilingData tiling;
    
    const gert::StorageShape* xShape = context->GetInputShape(0);
    uint32_t batchSize = xShape->GetStorageShape().GetDim(0);
    uint32_t numChannels = xShape->GetStorageShape().GetDim(1);
    uint32_t numHW = 1;
    for (int i = 2; i < xShape->GetStorageShape().GetDimNum(); i++) {
        numHW *= xShape->GetStorageShape().GetDim(i);
    }
    
    const auto* attrs = context->GetAttrs();
    uint32_t numGroups = *(attrs->GetAttrPointer<int32_t>(0));
    float eps = 1e-5f;
    const float* epsPtr = attrs->GetAttrPointer<float>(4);
    if (epsPtr) {
        eps = *epsPtr;
    }

    uint32_t channelsPerGroup = numChannels / numGroups;
    uint32_t groupSize = channelsPerGroup * numHW;
    
    // Each block processes one (batch, group) pair
    uint32_t totalTasks = batchSize * numGroups;
    uint32_t blockDim = totalTasks;
    if (blockDim > 256) blockDim = 256;
    
    context->SetBlockDim(blockDim);
    
    tiling.set_batchSize(batchSize);
    tiling.set_numGroups(numGroups);
    tiling.set_numChannels(numChannels);
    tiling.set_numHW(numHW);
    tiling.set_channelsPerGroup(channelsPerGroup);
    tiling.set_groupSize(groupSize);
    tiling.set_eps(eps);
    
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
class GroupNormCustom : public OpDef {
public:
    explicit GroupNormCustom(const char* name) : OpDef(name)
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
        this->Attr("num_groups").AttrType(REQUIRED).Int();
        this->Attr("num_channels").AttrType(REQUIRED).Int();
        this->Attr("num_hw").AttrType(REQUIRED).Int();
        this->Attr("batch_size").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-5f);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GroupNormCustom);
}
