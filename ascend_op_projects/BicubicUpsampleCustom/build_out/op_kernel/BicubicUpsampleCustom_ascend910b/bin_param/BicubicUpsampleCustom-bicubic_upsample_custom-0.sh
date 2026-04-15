#!/bin/bash
echo "[ascend910b] Generating BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e ..."
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
res=$(opc $1 --main_func=bicubic_upsample_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/BicubicUpsampleCustom/build_out/op_kernel/BicubicUpsampleCustom_ascend910b/bin_param/BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e.json ; then
  echo "$2/BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e.json not generated!"
  exit 1
fi

if ! test -f $2/BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e.o ; then
  echo "$2/BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating BicubicUpsampleCustom_d18af2390f7984ca5594cb2947453d9e Done"
