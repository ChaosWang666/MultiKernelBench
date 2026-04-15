#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ReluSelfAttentionCustom)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ReluSelfAttentionCustom);

}

#endif
