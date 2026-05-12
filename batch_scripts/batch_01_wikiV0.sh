#!/usr/bin/env bash
# Batch wikiV0 - 用 Claude Code CLI + 本地 wiki 检索生成 AscendC 算子
#
# 用法:
#   bash batch_scripts/batch_01_wikiV0.sh            # 默认: selected_ops.yaml 的 40 个 LLM 算子 (完整评测)
#   bash batch_scripts/batch_01_wikiV0.sh --smoke    # selected_ops_smoke.yaml 的 10 个 LLM 算子 (快速验证)
#
# 环境变量: STRATEGY (默认 add_shot, 已切到 anti-hack 加固版 prompt)
#           RUNS     (默认 1)
#           TIMEOUT  (默认 1200 秒)
set -euo pipefail

STRATEGY="${STRATEGY:-add_shot}"
RUNS="${RUNS:-1}"
TIMEOUT="${TIMEOUT:-1200}"

cd "$(dirname "$0")/.."

# ---- 解析子集选项 ----
OPS_SOURCE="selected"  # selected (40) | smoke (10)
for arg in "${@:-}"; do
    case "$arg" in
        ""|--selected) OPS_SOURCE="selected" ;;
        --smoke)       OPS_SOURCE="smoke" ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *) echo "[ERROR] 未知参数: $arg" >&2; exit 1 ;;
    esac
done

if [ "$OPS_SOURCE" = "smoke" ]; then
    YAML="selected_ops_smoke.yaml"
else
    YAML="selected_ops.yaml"
fi

ops_text="$(python - "$YAML" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
for op in data["ops"]:
    print(op["name"])
PY
)" || { echo "[ERROR] 读取 $YAML 失败" >&2; exit 1; }
readarray -t OPS <<< "$ops_text"

echo "==> Batch wikiV0 [${OPS_SOURCE}]: ${#OPS[@]} ops (from ${YAML})"
python generate_with_cc_and_wiki.py \
    --model-name claude-code-static \
    --strategy "$STRATEGY" \
    --runs "$RUNS" \
    --timeout "$TIMEOUT" \
    --with-wiki \
    --ops "${OPS[@]}"

echo "==> Batch wikiV0 [${OPS_SOURCE}] done."
