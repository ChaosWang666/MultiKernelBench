
#include "densenet121_dense_block_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 32;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Densenet121DenseBlockCustomTilingData tiling;
    uint32_t numLayers = static_cast<uint32_t>(context->GetAttrInt("num_layers"));
    uint32_t numInputFeatures = static_cast<uint32_t>(context->GetAttrInt("num_input_features"));
    uint32_t growthRate = static_cast<uint32_t>(context->GetAttrInt("growth_rate"));
    uint32_t batchSize = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetDim(0));
    uint32_t height = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetDim(2));
    uint32_t width = static_cast<uint32_t>(context->GetInputShape(0)->GetOriginShape().GetDim(3));
    uint32_t totalElements = batchSize * (numInputFeatures + numLayers * growthRate) * height * width;
    
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_numLayers(numLayers);
    tiling.set_numInputFeatures(numInputFeatures);
    tiling.set_growthRate(growthRate);
    tiling.set_batchSize(batchSize);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_totalElements(totalElements);
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
    uint32_t numLayers = static_cast<uint32_t>(context->GetAttrInt("num_layers"));
    uint32_t numInputFeatures = static_cast<uint32_t>(context->GetAttrInt("num_input_features"));
    uint32_t growthRate = static_cast<uint32_t>(context->GetAttrInt("growth_rate"));
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    y_shape->GetOriginShape().SetDim(1, numInputFeatures + numLayers * growthRate);
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
class Densenet121DenseBlockCustom : public OpDef {
public:
    explicit Densenet121DenseBlockCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_NCHW})
            .UnknownShapeFormat({ge::FORMAT_NCHW});
        this->Attr("num_layers").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("num_input_features").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->Attr("growth_rate").SetType(ATTR_TYPE_INT).SetDefault(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Densenet121DenseBlockCustom);
}
