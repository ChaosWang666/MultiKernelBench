#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(RmspropCustom)
    .INPUT(param, ge::TensorType::ALL())
    .INPUT(grad, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .OUTPUT(param_out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(RmspropCustom);

}

#endif
