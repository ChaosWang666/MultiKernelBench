#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GemmGroupNormSwishMultiplySwishCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(group_norm_weight, ge::TensorType::ALL())
    .INPUT(group_norm_bias, ge::TensorType::ALL())
    .INPUT(multiply_weight, ge::TensorType::ALL())
    .OUTPUT(z, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GemmGroupNormSwishMultiplySwishCustom);

}

#endif
