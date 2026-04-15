#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(multiplier, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom);

}

#endif
