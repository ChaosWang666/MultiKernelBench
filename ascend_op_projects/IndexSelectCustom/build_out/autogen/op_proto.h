#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(IndexSelectCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(indices, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(numRows, Int)
    .REQUIRED_ATTR(numCols, Int)
    .REQUIRED_ATTR(numIndices, Int)
    .OP_END_FACTORY_REG(IndexSelectCustom);

}

#endif
