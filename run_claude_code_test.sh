#!/usr/bin/env bash
# Run three Claude Code generation tests on AscendC fuse operators:
#   1. baseline (no extra context)
#   2. with ascendc-api-best-practices.md injected into the prompt
#   3. with local AscendC Kernel Wiki retrieval via Read/Glob/Grep
# All runs are evaluated via evaluation.py for compile/correctness rate.
#
# Env overrides:
#   CATEGORIES   space-separated category list (default: fuse)
#   STRATEGY     prompt strategy            (default: add_shot)
#   RUNS         repeat count               (default: 1)
#   TIMEOUT      per-op claude timeout (s)  (default: 600)
#   SKIP_BASE    set to 1 to skip baseline
#   SKIP_BP      set to 1 to skip best-practices run
#   SKIP_WIKI    set to 1 to skip wiki-retrieval run
#   SKIP_EVAL    set to 1 to skip evaluation.py invocations

set -euo pipefail

CATEGORIES="${CATEGORIES:-fuse}"
STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-600}"
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_BP="${SKIP_BP:-0}"
SKIP_WIKI="${SKIP_WIKI:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

cd "$(dirname "$0")"

run_eval() {
    local model_name="$1"
    if [[ "$SKIP_EVAL" == "1" ]]; then
        echo "[SKIP] evaluation.py for $model_name"
        return
    fi
    echo "==> Evaluating $model_name"
    python evaluation.py \
        --model "$model_name" \
        --strategy "$STRATEGY" \
        --categories $CATEGORIES \
        --runs "$RUNS"
}

if [[ "$SKIP_BASE" != "1" ]]; then
    echo "==> [Test 1] Baseline (no best-practices)"
    python generate_with_claude_code.py \
        --model-name claude-code \
        --strategy "$STRATEGY" \
        --categories $CATEGORIES \
        --runs "$RUNS" \
        --timeout "$TIMEOUT" \
        --disable-skills
    run_eval claude-code
else
    echo "[SKIP] baseline test"
fi

if [[ "$SKIP_BP" != "1" ]]; then
    echo "==> [Test 2] With ascendc-api-best-practices.md"
    python generate_with_claude_code.py \
        --model-name claude-code-bp \
        --strategy "$STRATEGY" \
        --categories $CATEGORIES \
        --runs "$RUNS" \
        --timeout "$TIMEOUT" \
        --disable-skills \
        --with-best-practices
    run_eval claude-code-bp
else
    echo "[SKIP] best-practices test"
fi

if [[ "$SKIP_WIKI" != "1" ]]; then
    echo "==> [Test 3] With AscendC Kernel Wiki retrieval"
    python generate_with_claude_code.py \
        --model-name claude-code-wiki \
        --strategy "$STRATEGY" \
        --categories $CATEGORIES \
        --runs "$RUNS" \
        --timeout "$TIMEOUT" \
        --disable-skills \
        --with-wiki
    run_eval claude-code-wiki
else
    echo "[SKIP] wiki test"
fi

echo "==> Done."
