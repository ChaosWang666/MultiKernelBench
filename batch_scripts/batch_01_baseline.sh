#!/usr/bin/env bash
# Batch 1/10 - Baseline (claude-code) - ops[0..9]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

echo "==> Batch 1/10 Baseline (claude-code): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --ops \
        convtranspose3d_relu_groupnorm \
        conv2d_subtract_hard_swish_max_pool_mish \
        conv_transpose3d_batch_norm_avg_pool_avg_pool \
        conv3d_divide_max_global_avg_pool_bias_add_sum \
        gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu \
        conv3d_hardswish_relu_softmax_mean \
        conv2d_min_add_multiply \
        conv_transpose2d_gelu_group_norm \
        conv_transpose2d_add_min_gelu_multiply \
        matmul_divide_gelu

echo "==> Batch 1/10 Baseline done."
