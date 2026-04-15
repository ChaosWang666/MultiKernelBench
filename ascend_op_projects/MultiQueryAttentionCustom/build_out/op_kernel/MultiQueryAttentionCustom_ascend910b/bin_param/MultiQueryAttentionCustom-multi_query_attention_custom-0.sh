#!/bin/bash
echo "[ascend910b] Generating MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a ..."
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
res=$(opc $1 --main_func=multi_query_attention_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MultiQueryAttentionCustom/build_out/op_kernel/MultiQueryAttentionCustom_ascend910b/bin_param/MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a.json ; then
  echo "$2/MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a.json not generated!"
  exit 1
fi

if ! test -f $2/MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a.o ; then
  echo "$2/MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MultiQueryAttentionCustom_414bae1155986a23db59e77395aa2d0a Done"
