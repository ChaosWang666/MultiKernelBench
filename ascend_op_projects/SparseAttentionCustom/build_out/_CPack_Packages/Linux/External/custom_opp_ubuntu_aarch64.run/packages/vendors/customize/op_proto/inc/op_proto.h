#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseAttentionCustom)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .REQUIRED_ATTR(batchSize, Int)
    .REQUIRED_ATTR(numHeads, Int)
    .REQUIRED_ATTR(seqLen, Int)
    .REQUIRED_ATTR(headDim, Int)
    .REQUIRED_ATTR(windowSize, Int)
    .OP_END_FACTORY_REG(SparseAttentionCustom);

}

#endif
