#!/bin/bash
echo "[ascend910b] Generating ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada ..."
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
res=$(opc $1 --main_func=product_reduction_over_a_dimension_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ProductReductionOverADimensionCustom/build_out/op_kernel/ProductReductionOverADimensionCustom_ascend910b/bin_param/ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada.json ; then
  echo "$2/ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada.json not generated!"
  exit 1
fi

if ! test -f $2/ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada.o ; then
  echo "$2/ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ProductReductionOverADimensionCustom_b97adbb741111e50fded5af9394ceada Done"
