#!/bin/bash
echo "[ascend910b] Generating ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee ..."
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
res=$(opc $1 --main_func=clamp_broadcast_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ClampBroadcastCustom/build_out/op_kernel/ClampBroadcastCustom_ascend910b/bin_param/ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee.json ; then
  echo "$2/ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee.json not generated!"
  exit 1
fi

if ! test -f $2/ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee.o ; then
  echo "$2/ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ClampBroadcastCustom_52e8c9e06fb1d96842a0c561781240ee Done"
