#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(TripletMarginLossCustom)
    .INPUT(anchor, ge::TensorType::ALL())
    .INPUT(positive, ge::TensorType::ALL())
    .INPUT(negative, ge::TensorType::ALL())
    .OUTPUT(loss, ge::TensorType::ALL())
    .ATTR(margin, Float, 1)
    .OP_END_FACTORY_REG(TripletMarginLossCustom);

}

#endif
