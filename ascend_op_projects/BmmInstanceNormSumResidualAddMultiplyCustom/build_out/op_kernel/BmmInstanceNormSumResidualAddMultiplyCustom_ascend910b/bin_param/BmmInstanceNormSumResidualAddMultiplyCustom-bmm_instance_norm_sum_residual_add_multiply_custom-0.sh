#!/bin/bash
echo "[ascend910b] Generating BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2 ..."
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
res=$(opc $1 --main_func=bmm_instance_norm_sum_residual_add_multiply_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/BmmInstanceNormSumResidualAddMultiplyCustom/build_out/op_kernel/BmmInstanceNormSumResidualAddMultiplyCustom_ascend910b/bin_param/BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2.json ; then
  echo "$2/BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2.json not generated!"
  exit 1
fi

if ! test -f $2/BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2.o ; then
  echo "$2/BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating BmmInstanceNormSumResidualAddMultiplyCustom_0b7379f1bff9596d646ee4dd0e29d7b2 Done"
