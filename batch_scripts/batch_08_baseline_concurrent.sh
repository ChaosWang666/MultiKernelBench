#!/usr/bin/env bash
# Batch 8/10 - Baseline (claude-code) - ops[70..79] - CONCURRENT GEN+EVAL
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}"

cd "$(dirname "$0")/.."

echo "==> Batch 8/10 Baseline concurrent (claude-code): 10 ops"
python generate_and_evaluate.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --disable-skills \
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

echo "==> Batch 8/10 Baseline concurrent done."
