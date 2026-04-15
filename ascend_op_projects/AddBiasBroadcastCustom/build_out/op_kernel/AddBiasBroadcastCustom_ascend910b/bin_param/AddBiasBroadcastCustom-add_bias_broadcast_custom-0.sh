#!/bin/bash
echo "[ascend910b] Generating AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684 ..."
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
res=$(opc $1 --main_func=add_bias_broadcast_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/AddBiasBroadcastCustom/build_out/op_kernel/AddBiasBroadcastCustom_ascend910b/bin_param/AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684.json ; then
  echo "$2/AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684.json not generated!"
  exit 1
fi

if ! test -f $2/AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684.o ; then
  echo "$2/AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684 Done"
