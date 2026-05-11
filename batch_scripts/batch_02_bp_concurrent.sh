#!/usr/bin/env bash
# Batch 2/10 - Best-practices (claude-code-bp) - ops[10..19] - CONCURRENT GEN+EVAL
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}"

cd "$(dirname "$0")/.."

echo "==> Batch 2/10 Best-practices concurrent (claude-code-bp): 10 ops"
python generate_and_evaluate.py \
    --model-name claude-code-bp \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --disable-skills \
    --with-best-practices \
    --ops \
        conv2d_relu_hard_swish \
        conv2d_tanh_scaling_bias_add_max \
        conv_transpose3d_multiply_max_global_avg_pool_clamp \
        matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp \
        gemm_divide_sum_scaling \
        conv2d_batch_norm_scaling \
        conv_transpose3d_avg_pool_clamp_softmax_multiply \
        conv_transpose2d_bias_add_clamp_scaling_clamp_divide \
        conv3d_multiply_instance_norm_clamp_multiply_max \
        conv_transpose3d_layer_norm_gelu_scaling

echo "==> Batch 2/10 Best-practices concurrent done."
