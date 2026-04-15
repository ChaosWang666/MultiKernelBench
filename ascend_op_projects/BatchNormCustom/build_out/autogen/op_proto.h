#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(BatchNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(running_mean, ge::TensorType::ALL())
    .INPUT(running_var, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(BatchNormCustom);

}

#endif
