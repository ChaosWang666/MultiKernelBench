#!/usr/bin/env bash
# Batch 5/10 - Baseline (claude-code) - ops[40..49] - CONCURRENT GEN+EVAL
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}"

cd "$(dirname "$0")/.."

echo "==> Batch 5/10 Baseline concurrent (claude-code): 10 ops"
python generate_and_evaluate.py \
    --model-name claude-code \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --disable-skills \
    --ops \
        gemm_swish_divide_clamp_tanh_clamp \
        convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply \
        conv2d_divide_leaky_relu \
        matmul_dropout_mean_softmax \
        conv_transpose3d_swish_group_norm_hard_swish \
        conv2d_instance_norm_divide \
        conv2d_scaling_min \
        conv_transpose3d_scaling_avg_pool_bias_add_scaling \
        gemm_max_subtract_gelu \
        gemm_batch_norm_scaling_softmax

echo "==> Batch 5/10 Baseline concurrent done."
