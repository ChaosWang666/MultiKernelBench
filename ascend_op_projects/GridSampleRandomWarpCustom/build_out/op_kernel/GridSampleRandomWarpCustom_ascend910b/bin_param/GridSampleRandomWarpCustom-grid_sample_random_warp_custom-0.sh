#!/bin/bash
echo "[ascend910b] Generating GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951 ..."
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
res=$(opc $1 --main_func=grid_sample_random_warp_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GridSampleRandomWarpCustom/build_out/op_kernel/GridSampleRandomWarpCustom_ascend910b/bin_param/GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951.json ; then
  echo "$2/GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951.json not generated!"
  exit 1
fi

if ! test -f $2/GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951.o ; then
  echo "$2/GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GridSampleRandomWarpCustom_c9f0545f31d0a5257e622b0f9fcf4951 Done"
