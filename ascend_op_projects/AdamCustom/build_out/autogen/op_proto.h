#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AdamCustom)
    .INPUT(param, ge::TensorType::ALL())
    .INPUT(grad, ge::TensorType::ALL())
    .INPUT(m, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .OUTPUT(param_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(beta1, Float)
    .REQUIRED_ATTR(beta2, Float)
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(eps, Float)
    .REQUIRED_ATTR(step, Int)
    .OP_END_FACTORY_REG(AdamCustom);

}

#endif
