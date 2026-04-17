#!/usr/bin/env bash
# Batch 4/10 - Best-practices (claude-code-bp) - ops[30..39]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-600}"

cd "$(dirname "$0")/.."

echo "==> Batch 4/10 Best-practices (claude-code-bp): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code-bp \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --with-best-practices \
    --ops \
        matmul_swish_sum_group_norm \
        conv3d_max_log_sum_exp_relu \
        conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp \
        conv_transpose3d_sum_layer_norm_avg_pool_gelu \
        conv2d_avg_pool_sigmoid_sum \
        conv3d_relu_leaky_relu_gelu_sigmoid_bias_add \
        gemm_group_norm_min_bias_add \
        conv3d_softmax_max_pool_max_pool \
        gemm_group_norm_hardtanh \
        conv2d_group_norm_scale_max_pool_clamp

echo "==> Batch 4/10 Best-practices done."
