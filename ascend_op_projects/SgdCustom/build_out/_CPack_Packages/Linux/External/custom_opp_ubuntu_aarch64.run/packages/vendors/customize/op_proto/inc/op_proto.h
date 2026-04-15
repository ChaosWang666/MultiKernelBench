#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SgdCustom)
    .INPUT(param, ge::TensorType::ALL())
    .INPUT(grad, ge::TensorType::ALL())
    .INPUT(velocity, ge::TensorType::ALL())
    .OUTPUT(param_out, ge::TensorType::ALL())
    .ATTR(momentum, Float, 0.9)
    .ATTR(lr, Float, 0.01)
    .OP_END_FACTORY_REG(SgdCustom);

}

#endif
