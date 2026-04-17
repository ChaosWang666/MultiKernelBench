#!/usr/bin/env bash
# Batch 9/10 - Best-practices (claude-code-bp) - ops[80..89]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-600}"

cd "$(dirname "$0")/.."

echo "==> Batch 9/10 Best-practices (claude-code-bp): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code-bp \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --with-best-practices \
    --ops \
        gemm_bias_add_hardtanh_mish_group_norm \
        conv3d_scaling_tanh_multiply_sigmoid \
        conv_transpose3d_softmax_sigmoid \
        matmul_gelu_softmax \
        conv2d_add_scale_sigmoid_group_norm \
        matmul_avg_pool_gelu_scale_max \
        convtranspose2d_softmax_biasadd_scaling_sigmoid \
        matmul_mish_mish \
        matmul_max_pool_sum_scale \
        conv_transpose3d_leaky_relu_multiply_leaky_relu_max

echo "==> Batch 9/10 Best-practices done."
