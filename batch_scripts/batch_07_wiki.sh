#!/usr/bin/env bash
# Batch 7/10 - Wiki (claude-code-wiki) - ops[60..69]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

echo "==> Batch 7/10 Wiki (claude-code-wiki): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code-wiki \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --with-wiki \
    --ops \
        bmm_instance_norm_sum_residual_add_multiply \
        conv_transpose3d_max_pool_softmax_subtract_swish_max \
        conv2d_mish_mish \
        gemm_add_relu \
        gemm_relu_divide \
        conv3d_leaky_relu_sum_clamp_gelu \
        matmul_group_norm_leaky_relu_sum \
        conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max \
        gemm_sigmoid_sum_log_sum_exp \
        conv_transpose3d_add_hard_swish

echo "==> Batch 7/10 Wiki done."
