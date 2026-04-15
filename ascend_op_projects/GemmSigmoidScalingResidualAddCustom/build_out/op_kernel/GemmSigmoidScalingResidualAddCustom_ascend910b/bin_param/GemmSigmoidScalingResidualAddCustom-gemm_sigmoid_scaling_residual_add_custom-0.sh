#!/bin/bash
echo "[ascend910b] Generating GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031 ..."
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
res=$(opc $1 --main_func=gemm_sigmoid_scaling_residual_add_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GemmSigmoidScalingResidualAddCustom/build_out/op_kernel/GemmSigmoidScalingResidualAddCustom_ascend910b/bin_param/GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031.json ; then
  echo "$2/GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031.json not generated!"
  exit 1
fi

if ! test -f $2/GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031.o ; then
  echo "$2/GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GemmSigmoidScalingResidualAddCustom_11afd12da0eb929e4db2bb7a0ce66031 Done"
