#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GruHiddenCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(hx, ge::TensorType::ALL())
    .INPUT(w_ih, ge::TensorType::ALL())
    .INPUT(w_hh, ge::TensorType::ALL())
    .INPUT(b_ih, ge::TensorType::ALL())
    .INPUT(b_hh, ge::TensorType::ALL())
    .OUTPUT(hy, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GruHiddenCustom);

}

#endif
