#!/usr/bin/env bash
# Batch 7/10 - Baseline (claude-code) - ops[60..69] - CONCURRENT GEN+EVAL
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}"

cd "$(dirname "$0")/.."

echo "==> Batch 7/10 Baseline concurrent (claude-code): 10 ops"
python generate_and_evaluate.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --disable-skills \
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

echo "==> Batch 7/10 Baseline concurrent done."
