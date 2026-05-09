export ASCEND_RT_VISIBLE_DEVICES=15

model_name="claude-code-static"     # 配置类似 output/ascendc/add_shot/0.0-1.0/claude-code-wiki 最后一级目录名
STRATEGY="${STRATEGY:-add_shot}"
CATEGORIES="${CATEGORIES:-fuse}"    # 测fuse类的
RUNS="${RUNS:-1}"                   # 跑一次，看pass@1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_EXT_DIR="${SCRIPT_DIR}/ascend_op_projects/CppExtension"
CONFIG_PY="${SCRIPT_DIR}/config.py"

# 自动探测当前 NPU 的 SOC（A2: Ascend910B2 等；A3: Ascend910_93xx），
# 同步写回 config.py 的 ascendc_device，避免按错误 soc 编译导致运行时
# "binary_info_config.json of socVersion [...] does not support opType ..."
DETECTED_SOC="$(ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" python3 - <<'PY' 2>/dev/null
try:
    import acl
    name = acl.get_soc_name()
    if name:
        print(name)
except Exception:
    pass
PY
)"

if [ -n "$DETECTED_SOC" ]; then
    DESIRED_DEVICE="ai_core-${DETECTED_SOC}"
    CURRENT_DEVICE="$(sed -n "s/^ascendc_device = '\(.*\)'$/\1/p" "${CONFIG_PY}")"
    if [ "$DESIRED_DEVICE" != "$CURRENT_DEVICE" ]; then
        echo "[run_cc_static.sh] SOC changed: '${CURRENT_DEVICE}' -> '${DESIRED_DEVICE}', updating config.py and cleaning stale per-soc artifacts"
        sed -i "s|^ascendc_device = .*|ascendc_device = '${DESIRED_DEVICE}'|" "${CONFIG_PY}"
        # 已部署的 opp 包和此前为旧 soc 编译的算子工程目录都得清掉，
        # 否则 aclnn 运行时仍会用旧 soc 的 binary_info_config.json
        rm -rf "${SCRIPT_DIR}/ascend_op_projects/opp"
        find "${SCRIPT_DIR}/ascend_op_projects" -mindepth 1 -maxdepth 1 -type d ! -name CppExtension -exec rm -rf {} +
    else
        echo "[run_cc_static.sh] Detected SOC ${DETECTED_SOC}, config.py already matches"
    fi

    # 把 SOC 名映射到 opbuild/cmake 用的 soc 家族（platform_config 里那一份）。
    # 只覆盖目前真在用的 A2/A3 家族；其它芯片可按需补充。
    case "$DETECTED_SOC" in
        Ascend910_93*) DESIRED_FAMILY=ascend910_93 ;;
        Ascend910_95*) DESIRED_FAMILY=ascend910_95 ;;
        Ascend910_55*) DESIRED_FAMILY=ascend910_55 ;;
        Ascend910_96*) DESIRED_FAMILY=ascend910_96 ;;
        Ascend910B*)   DESIRED_FAMILY=ascend910b ;;
        Ascend910A*)   DESIRED_FAMILY=ascend910 ;;
        Ascend310P*)   DESIRED_FAMILY=ascend310p ;;
        Ascend310B*)   DESIRED_FAMILY=ascend310b ;;
        *)             DESIRED_FAMILY="" ;;
    esac

    # LLM 是按 A2 的 few-shot 学的，op_host 里写死 AICore().AddConfig("ascend910b")。
    # opbuild 直接读 host 源决定生成哪份 aic-<family>-ops-info.ini，写错了
    # kernel 编译目标根本不会触发，整个 binary 阶段会静默跳过。换 soc 时把所有
    # 已生成 .txt 里的 AddConfig 串规范化到当前 family；A2 重跑时也能反向归位。
    if [ -n "$DESIRED_FAMILY" ] && [ -d "${SCRIPT_DIR}/output" ]; then
        echo "[run_cc_static.sh] Normalizing AddConfig(...) in generated .txt to '${DESIRED_FAMILY}'"
        find "${SCRIPT_DIR}/output" -type f -name "*.txt" \
            -exec sed -i "s|AddConfig(\"ascend910[^\"]*\")|AddConfig(\"${DESIRED_FAMILY}\")|g" {} +
    fi
else
    echo "[run_cc_static.sh] WARNING: failed to detect SOC via acl.get_soc_name(), leaving config.py unchanged"
fi

# 清理 CppExtension 残留产物：dist/ 中的旧 wheel（如 cp313）会导致
# `pip install *.whl` 在当前 cp311 环境下失败，使每个算子的 pybind 阶段失败
rm -rf "${CPP_EXT_DIR}/dist" "${CPP_EXT_DIR}/build" "${CPP_EXT_DIR}/custom_ops.egg-info"

python evaluation.py \
        --model "$model_name" \
        --strategy "$STRATEGY" \
        --categories $CATEGORIES \
        --runs "$RUNS"