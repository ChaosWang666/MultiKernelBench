#!/bin/bash
echo "[ascend910b] Generating WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11 ..."
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
res=$(opc $1 --main_func=where_broadcast_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/WhereBroadcastCustom/build_out/op_kernel/WhereBroadcastCustom_ascend910b/bin_param/WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11.json ; then
  echo "$2/WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11.json not generated!"
  exit 1
fi

if ! test -f $2/WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11.o ; then
  echo "$2/WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating WhereBroadcastCustom_52878de09d2b11ad43aaa03fbdc53d11 Done"
