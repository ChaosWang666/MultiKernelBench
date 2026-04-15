
#include "matmul_group_norm_leaky_relu_sum_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MatmulGroupNormLeakyReluSumCustomTilingData tiling;
    
    const auto attrs = context->GetAttrs();
    uint32_t batchSize = *(attrs->GetAttrPointer<int>(0));
    uint32_t inputSize = *(attrs->GetAttrPointer<int>(1));
    uint32_t hiddenSize = *(attrs->GetAttrPointer<int>(2));
    uint32_t numGroups = *(attrs->GetAttrPointer<int>(3));
    float eps = *(attrs->GetAttrPointer<float>(4));
    float negativeSlope = *(attrs->GetAttrPointer<float>(5));
    
    tiling.set_batchSize(batchSize);
    tiling.set_inputSize(inputSize);
    tiling.set_hiddenSize(hiddenSize);
    tiling.set_numGroups(numGroups);
    tiling.set_eps(eps);
    tiling.set_negativeSlope(negativeSlope);
    
    context->SetBlockDim(batchSize);
    
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
    gert::Shape* z_shape = context->GetOutputShape(0);
    // output shape is (batchSize, hiddenSize)
    // We get batchSize from x dim 0, hiddenSize from weight dim 0
    const gert::Shape* w_shape = context->GetInputShape(1);
    std::vector<int64_t> dims = {x_shape->GetDim(0), w_shape->GetDim(0)};
    *z_shape = gert::Shape(dims.size(), dims.data());
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
class MatmulGroupNormLeakyReluSumCustom : public OpDef {
public:
    explicit MatmulGroupNormLeakyReluSumCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
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
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("batchSize").AttrType(REQUIRED).Int();
        this->Attr("inputSize").AttrType(REQUIRED).Int();
        this->Attr("hiddenSize").AttrType(REQUIRED).Int();
        this->Attr("numGroups").AttrType(REQUIRED).Int();
        this->Attr("eps").AttrType(REQUIRED).Float();
        this->Attr("negativeSlope").AttrType(REQUIRED).Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(MatmulGroupNormLeakyReluSumCustom);
}
