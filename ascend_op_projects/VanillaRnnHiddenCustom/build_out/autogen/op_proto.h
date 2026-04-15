#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(VanillaRnnHiddenCustom)
    .INPUT(input, ge::TensorType::ALL())
    .INPUT(hidden, ge::TensorType::ALL())
    .INPUT(weight_ih, ge::TensorType::ALL())
    .INPUT(bias_ih, ge::TensorType::ALL())
    .OUTPUT(hidden_new, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(VanillaRnnHiddenCustom);

}

#endif
