#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(RmsNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_features, Int)
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(RmsNormCustom);

}

#endif
