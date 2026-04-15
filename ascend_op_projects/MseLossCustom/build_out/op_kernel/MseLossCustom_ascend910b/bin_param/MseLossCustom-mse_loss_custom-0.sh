#!/bin/bash
echo "[ascend910b] Generating MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d ..."
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
res=$(opc $1 --main_func=mse_loss_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MseLossCustom/build_out/op_kernel/MseLossCustom_ascend910b/bin_param/MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d.json ; then
  echo "$2/MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d.json not generated!"
  exit 1
fi

if ! test -f $2/MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d.o ; then
  echo "$2/MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MseLossCustom_2c3d3b1d52376dd6e9376f8f6478c63d Done"
