#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_groups, Int)
    .REQUIRED_ATTR(num_channels, Int)
    .REQUIRED_ATTR(num_hw, Int)
    .REQUIRED_ATTR(batch_size, Int)
    .ATTR(eps, Float, 1e-05)
    .OP_END_FACTORY_REG(GroupNormCustom);

}

#endif
