#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ConvTranspose2dMishAddHardtanhScalingCustom)
    .INPUT(x, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(add_value, Float)
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(ConvTranspose2dMishAddHardtanhScalingCustom);

}

#endif
