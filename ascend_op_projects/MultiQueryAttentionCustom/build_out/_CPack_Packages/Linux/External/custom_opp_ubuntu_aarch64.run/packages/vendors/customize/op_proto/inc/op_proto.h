#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MultiQueryAttentionCustom)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k, ge::TensorType::ALL())
    .INPUT(v, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .REQUIRED_ATTR(batch_size, Int)
    .REQUIRED_ATTR(seq_len, Int)
    .REQUIRED_ATTR(num_heads, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .OP_END_FACTORY_REG(MultiQueryAttentionCustom);

}

#endif
