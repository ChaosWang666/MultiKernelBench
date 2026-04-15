#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MseLossCustom)
    .INPUT(predictions, ge::TensorType::ALL())
    .INPUT(targets, ge::TensorType::ALL())
    .OUTPUT(result, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MseLossCustom);

}

#endif
