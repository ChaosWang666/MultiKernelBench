#!/usr/bin/env bash
# Batch 8/10 - Best-practices (claude-code-bp) - ops[70..79]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

echo "==> Batch 8/10 Best-practices (claude-code-bp): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code-bp \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --with-best-practices \
    --ops \
        conv2d_min_tanh_tanh \
        conv2d_relu_bias_add \
        gemm_scale_batchnorm \
        gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add \
        matmul_add_swish_tanh_gelu_hardtanh \
        matmul_sigmoid_sum \
        conv3d_mish_tanh \
        matmul_batch_norm_bias_add_divide_swish \
        conv2d_subtract_subtract_mish \
        conv_transpose3d_scale_batch_norm_global_avg_pool

echo "==> Batch 8/10 Best-practices done."
