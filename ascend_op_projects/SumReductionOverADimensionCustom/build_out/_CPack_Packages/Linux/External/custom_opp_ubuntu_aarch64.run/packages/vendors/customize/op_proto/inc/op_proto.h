#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SumReductionOverADimensionCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(shape0, Int)
    .REQUIRED_ATTR(shape1, Int)
    .REQUIRED_ATTR(shape2, Int)
    .OP_END_FACTORY_REG(SumReductionOverADimensionCustom);

}

#endif
