
#include "conv2d_instance_norm_divide_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 20;
const uint32_t TILE_SIZE = 4096;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Conv2dInstanceNormDivideCustomTilingData tiling;
    auto shape = context->GetInputShape(0)->GetOriginShape();
    uint32_t N = static_cast<uint32_t>(shape.GetDim(0));
    uint32_t C = static_cast<uint32_t>(shape.GetDim(1));
    uint32_t H = static_cast<uint32_t>(shape.GetDim(2));
    uint32_t W = static_cast<uint32_t>(shape.GetDim(3));

    uint32_t totalInstances = N * C;
    uint32_t hw = H * W;

    uint32_t blockDim = BLOCK_DIM;
    if (totalInstances < blockDim) {
        blockDim = totalInstances;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }
    uint32_t instancesPerCore = (totalInstances + blockDim - 1) / blockDim;

    auto attrs = context->GetAttrs();
    const float* dividePtr = attrs->GetAttrPointer<float>(0);
    float divideBy = (dividePtr != nullptr) ? (*dividePtr) : 1.0f;

    context->SetBlockDim(blockDim);
    tiling.set_totalInstances(totalInstances);
    tiling.set_hw(hw);
    tiling.set_tileSize(TILE_SIZE);
    tiling.set_instancesPerCore(instancesPerCore);
    tiling.set_eps(1e-5f);
    tiling.set_divideBy(divideBy);

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
class Conv2dInstanceNormDivideCustom : public OpDef {
public:
    explicit Conv2dInstanceNormDivideCustom(const char* name) : OpDef(name)
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
        this->Attr("divide_by").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Conv2dInstanceNormDivideCustom);
}
