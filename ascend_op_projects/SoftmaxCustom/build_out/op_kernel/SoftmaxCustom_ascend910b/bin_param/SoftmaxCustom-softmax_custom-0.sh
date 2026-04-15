#!/bin/bash
echo "[ascend910b] Generating SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c ..."
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
res=$(opc $1 --main_func=softmax_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/SoftmaxCustom/build_out/op_kernel/SoftmaxCustom_ascend910b/bin_param/SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c.json ; then
  echo "$2/SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c.json not generated!"
  exit 1
fi

if ! test -f $2/SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c.o ; then
  echo "$2/SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating SoftmaxCustom_a67ad9463e03d331bfbcaed13beb864c Done"
