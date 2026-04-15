#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LambCustom)
    .INPUT(param, ge::TensorType::ALL())
    .INPUT(m, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(LambCustom);

}

#endif
