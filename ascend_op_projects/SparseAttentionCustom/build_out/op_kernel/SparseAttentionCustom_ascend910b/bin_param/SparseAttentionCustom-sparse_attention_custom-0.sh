#!/bin/bash
echo "[ascend910b] Generating SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8 ..."
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
res=$(opc $1 --main_func=sparse_attention_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SparseAttentionCustom/build_out/op_kernel/SparseAttentionCustom_ascend910b/bin_param/SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8.json ; then
  echo "$2/SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8.json not generated!"
  exit 1
fi

if ! test -f $2/SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8.o ; then
  echo "$2/SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating SparseAttentionCustom_9968fb60104f267e08e032041b93e1d8 Done"
