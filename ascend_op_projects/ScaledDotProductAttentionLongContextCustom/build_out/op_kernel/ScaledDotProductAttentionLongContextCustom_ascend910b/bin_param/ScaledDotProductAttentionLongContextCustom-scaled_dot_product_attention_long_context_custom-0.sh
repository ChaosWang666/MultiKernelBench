#!/bin/bash
echo "[ascend910b] Generating ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822 ..."
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
res=$(opc $1 --main_func=scaled_dot_product_attention_long_context_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ScaledDotProductAttentionLongContextCustom/build_out/op_kernel/ScaledDotProductAttentionLongContextCustom_ascend910b/bin_param/ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822.json ; then
  echo "$2/ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822.json not generated!"
  exit 1
fi

if ! test -f $2/ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822.o ; then
  echo "$2/ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ScaledDotProductAttentionLongContextCustom_da733d4770a06e44447f6fd9d3f58822 Done"
