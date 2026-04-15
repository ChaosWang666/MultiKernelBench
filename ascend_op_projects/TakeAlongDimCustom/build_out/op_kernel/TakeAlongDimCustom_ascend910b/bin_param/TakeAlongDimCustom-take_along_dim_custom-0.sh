#!/bin/bash
echo "[ascend910b] Generating TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8 ..."
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
res=$(opc $1 --main_func=take_along_dim_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/TakeAlongDimCustom/build_out/op_kernel/TakeAlongDimCustom_ascend910b/bin_param/TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8.json ; then
  echo "$2/TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8.json not generated!"
  exit 1
fi

if ! test -f $2/TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8.o ; then
  echo "$2/TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating TakeAlongDimCustom_60fabbe96ab99be8ac8d10fbe509f8b8 Done"
