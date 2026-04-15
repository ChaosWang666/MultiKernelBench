#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LstmCnCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(h0, ge::TensorType::ALL())
    .INPUT(c0, ge::TensorType::ALL())
    .OUTPUT(hn, ge::TensorType::ALL())
    .OUTPUT(cn, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(LstmCnCustom);

}

#endif
