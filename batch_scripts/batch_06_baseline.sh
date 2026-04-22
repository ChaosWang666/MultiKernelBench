#!/usr/bin/env bash
# Batch 6/10 - Baseline (claude-code) - ops[50..59]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

echo "==> Batch 6/10 Baseline (claude-code): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --ops \
        conv2d_multiply_leaky_relu_gelu \
        conv_transpose3d_batch_norm_subtract \
        convtranspose2d_batchnorm_tanh_maxpool_groupnorm \
        conv2d_activation_batch_norm \
        gemm_scale_batch_norm \
        conv_transpose3d_sum_residual_add_multiply_residual_add \
        conv_transpose3d_clamp_min_divide \
        gemm_scaling_hard_tanh_gelu \
        matmul_scale_residual_add_clamp_log_sum_exp_mish \
        matmul_scaling_residual_add

echo "==> Batch 6/10 Baseline done."
