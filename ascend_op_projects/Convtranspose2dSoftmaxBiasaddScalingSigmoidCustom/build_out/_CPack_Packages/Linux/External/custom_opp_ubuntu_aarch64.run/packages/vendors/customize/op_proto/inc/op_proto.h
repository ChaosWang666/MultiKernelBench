#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .ATTR(scaling_factor, Float, 1)
    .OP_END_FACTORY_REG(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom);

}

#endif
