#!/bin/bash
echo "[ascend910b] Generating MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126 ..."
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
res=$(opc $1 --main_func=mean_reduction_over_a_dimension_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MeanReductionOverADimensionCustom/build_out/op_kernel/MeanReductionOverADimensionCustom_ascend910b/bin_param/MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126.json ; then
  echo "$2/MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126.json not generated!"
  exit 1
fi

if ! test -f $2/MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126.o ; then
  echo "$2/MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MeanReductionOverADimensionCustom_ab319dfe2d5a0d14b7bad2c7364bb126 Done"
