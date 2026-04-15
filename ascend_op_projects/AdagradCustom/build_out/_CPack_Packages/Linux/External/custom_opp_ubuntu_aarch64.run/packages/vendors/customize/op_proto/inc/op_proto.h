#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AdagradCustom)
    .INPUT(param, ge::TensorType::ALL())
    .INPUT(grad, ge::TensorType::ALL())
    .INPUT(accum, ge::TensorType::ALL())
    .OUTPUT(param_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(lr, Float)
    .REQUIRED_ATTR(eps, Float)
    .OP_END_FACTORY_REG(AdagradCustom);

}

#endif
