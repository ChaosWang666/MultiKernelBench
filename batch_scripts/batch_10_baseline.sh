#!/usr/bin/env bash
# Batch 10/10 - Baseline (claude-code) - ops[90..99]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-600}"

cd "$(dirname "$0")/.."

echo "==> Batch 10/10 Baseline (claude-code): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --ops \
        gemm_multiply_leakyrelu \
        gemm_sigmoid_scaling_residual_add \
        conv_transpose2d_mish_add_hardtanh_scaling \
        gemm_batch_norm_gelu_group_norm_mean_relu \
        conv3d_group_norm_mean \
        conv2d_hard_swish_relu \
        convtranspose3d_mean_add_softmax_tanh_scaling \
        conv_transpose3d_max_max_sum \
        conv_transpose2d_min_sum_gelu_add \
        conv3d_min_softmax

echo "==> Batch 10/10 Baseline done."
