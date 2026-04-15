#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MatmulSwishSumGroupNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(groupNormWeight, ge::TensorType::ALL())
    .INPUT(groupNormBias, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MatmulSwishSumGroupNormCustom);

}

#endif
