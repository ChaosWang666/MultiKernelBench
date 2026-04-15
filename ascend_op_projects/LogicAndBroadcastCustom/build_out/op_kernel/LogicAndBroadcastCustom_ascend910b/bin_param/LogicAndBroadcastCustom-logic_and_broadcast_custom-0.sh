#!/bin/bash
echo "[ascend910b] Generating LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722 ..."
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
res=$(opc $1 --main_func=logic_and_broadcast_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/LogicAndBroadcastCustom/build_out/op_kernel/LogicAndBroadcastCustom_ascend910b/bin_param/LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722.json ; then
  echo "$2/LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722.json not generated!"
  exit 1
fi

if ! test -f $2/LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722.o ; then
  echo "$2/LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating LogicAndBroadcastCustom_a16b9c309eef7ef48d55b76260046722 Done"
