#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MiniGptBlockCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(ln1_weight, ge::TensorType::ALL())
    .INPUT(ln1_bias, ge::TensorType::ALL())
    .INPUT(attn_c_attn_weight, ge::TensorType::ALL())
    .INPUT(attn_c_attn_bias, ge::TensorType::ALL())
    .INPUT(attn_c_proj_weight, ge::TensorType::ALL())
    .INPUT(attn_c_proj_bias, ge::TensorType::ALL())
    .INPUT(ln2_weight, ge::TensorType::ALL())
    .INPUT(ln2_bias, ge::TensorType::ALL())
    .INPUT(mlp_c_fc_weight, ge::TensorType::ALL())
    .INPUT(mlp_c_fc_bias, ge::TensorType::ALL())
    .INPUT(mlp_c_proj_weight, ge::TensorType::ALL())
    .INPUT(mlp_c_proj_bias, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MiniGptBlockCustom);

}

#endif
