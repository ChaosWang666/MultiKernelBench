#!/bin/bash
echo "[ascend910b] Generating Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f ..."
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
res=$(opc $1 --main_func=conv2d_add_scale_sigmoid_group_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/Conv2dAddScaleSigmoidGroupNormCustom/build_out/op_kernel/Conv2dAddScaleSigmoidGroupNormCustom_ascend910b/bin_param/Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f.json ; then
  echo "$2/Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f.json not generated!"
  exit 1
fi

if ! test -f $2/Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f.o ; then
  echo "$2/Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating Conv2dAddScaleSigmoidGroupNormCustom_5b8d8a1ed63f3eb5febfe82860100c4f Done"
