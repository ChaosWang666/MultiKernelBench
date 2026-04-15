#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GemmScaleBatchNormCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(weight, ge::TensorType::ALL())
    .INPUT(bias, ge::TensorType::ALL())
    .INPUT(scale, ge::TensorType::ALL())
    .INPUT(mean, ge::TensorType::ALL())
    .INPUT(variance, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GemmScaleBatchNormCustom);

}

#endif
