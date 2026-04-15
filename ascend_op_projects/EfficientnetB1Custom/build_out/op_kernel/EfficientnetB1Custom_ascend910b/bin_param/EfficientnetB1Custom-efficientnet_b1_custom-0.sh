#!/bin/bash
echo "[ascend910b] Generating EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7 ..."
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
res=$(opc $1 --main_func=efficientnet_b1_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/EfficientnetB1Custom/build_out/op_kernel/EfficientnetB1Custom_ascend910b/bin_param/EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7.json ; then
  echo "$2/EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7.json not generated!"
  exit 1
fi

if ! test -f $2/EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7.o ; then
  echo "$2/EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating EfficientnetB1Custom_9813dd786bbe6a842f74c24a09fd0ce7 Done"
