#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SwinMlpCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(fc1_weight, ge::TensorType::ALL())
    .INPUT(fc1_bias, ge::TensorType::ALL())
    .INPUT(fc2_weight, ge::TensorType::ALL())
    .INPUT(fc2_bias, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(SwinMlpCustom);

}

#endif
