#!/usr/bin/env bash
# Batch 3/10 - Wiki (claude-code-wiki) - ops[20..29]
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

echo "==> Batch 3/10 Wiki (claude-code-wiki): 10 ops"
python generate_with_claude_code.py \
    --model-name claude-code-wiki \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --disable-skills \
    --with-wiki \
    --ops \
        conv3d_group_norm_min_clamp_dropout \
        gemm_group_norm_swish_multiply_swish \
        conv2d_subtract_tanh_subtract_avg_pool \
        matmul_swish_scaling \
        conv2d_gelu_global_avg_pool \
        matmul_min_subtract \
        conv_transpose2d_multiply_global_avg_pool_global_avg_pool_mean \
        conv_transpose2d_max_pool_hardtanh_mean_tanh \
        matmul_subtract_multiply_relu \
        conv_transpose2d_subtract_tanh

echo "==> Batch 3/10 Wiki done."
