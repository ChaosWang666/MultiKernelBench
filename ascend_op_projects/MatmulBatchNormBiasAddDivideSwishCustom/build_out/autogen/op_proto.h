#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MatmulBatchNormBiasAddDivideSwishCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .REQUIRED_ATTR(divide_value, Float)
    .OP_END_FACTORY_REG(MatmulBatchNormBiasAddDivideSwishCustom);

}

#endif
