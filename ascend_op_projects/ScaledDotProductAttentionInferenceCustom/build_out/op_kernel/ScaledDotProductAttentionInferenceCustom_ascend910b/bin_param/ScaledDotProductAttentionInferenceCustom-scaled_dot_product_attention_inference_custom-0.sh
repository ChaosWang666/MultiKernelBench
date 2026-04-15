#!/bin/bash
echo "[ascend910b] Generating ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7 ..."
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(opc $1 --main_func=scaled_dot_product_attention_inference_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ScaledDotProductAttentionInferenceCustom/build_out/op_kernel/ScaledDotProductAttentionInferenceCustom_ascend910b/bin_param/ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7.json ; then
  echo "$2/ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7.json not generated!"
  exit 1
fi

if ! test -f $2/ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7.o ; then
  echo "$2/ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ScaledDotProductAttentionInferenceCustom_1cd6c4d0344bece9bc0c30e5ff88f0c7 Done"
