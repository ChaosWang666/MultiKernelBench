#!/bin/bash
echo "[ascend910b] Generating GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d ..."
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
res=$(opc $1 --main_func=gemm_group_norm_min_bias_add_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GemmGroupNormMinBiasAddCustom/build_out/op_kernel/GemmGroupNormMinBiasAddCustom_ascend910b/bin_param/GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d.json ; then
  echo "$2/GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d.json not generated!"
  exit 1
fi

if ! test -f $2/GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d.o ; then
  echo "$2/GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GemmGroupNormMinBiasAddCustom_e382995d8d649738111c5543faa71a1d Done"
