#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GemmSigmoidScalingResidualAddCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(z, ge::TensorType::ALL())
    .REQUIRED_ATTR(scalingFactor, Float)
    .OP_END_FACTORY_REG(GemmSigmoidScalingResidualAddCustom);

}

#endif
