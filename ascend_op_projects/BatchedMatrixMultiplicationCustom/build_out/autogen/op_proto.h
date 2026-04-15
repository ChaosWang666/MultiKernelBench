#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(BatchedMatrixMultiplicationCustom)
    .INPUT(a, ge::TensorType::ALL())
    .INPUT(b, ge::TensorType::ALL())
    .OUTPUT(c, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(BatchedMatrixMultiplicationCustom);

}

#endif
