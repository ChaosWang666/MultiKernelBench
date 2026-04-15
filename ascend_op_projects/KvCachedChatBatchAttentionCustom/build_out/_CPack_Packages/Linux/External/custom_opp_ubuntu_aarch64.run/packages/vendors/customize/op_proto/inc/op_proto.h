#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(KvCachedChatBatchAttentionCustom)
    .INPUT(q, ge::TensorType::ALL())
    .INPUT(k_cache, ge::TensorType::ALL())
    .INPUT(v_cache, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(KvCachedChatBatchAttentionCustom);

}

#endif
