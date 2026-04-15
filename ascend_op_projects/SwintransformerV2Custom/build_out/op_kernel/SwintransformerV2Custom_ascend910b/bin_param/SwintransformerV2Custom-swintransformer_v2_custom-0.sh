#!/bin/bash
echo "[ascend910b] Generating SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776 ..."
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
res=$(opc $1 --main_func=swintransformer_v2_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SwintransformerV2Custom/build_out/op_kernel/SwintransformerV2Custom_ascend910b/bin_param/SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776.json ; then
  echo "$2/SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776.json not generated!"
  exit 1
fi

if ! test -f $2/SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776.o ; then
  echo "$2/SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating SwintransformerV2Custom_8738b66f6bb91e4a706b5a63316f1776 Done"
