#!/bin/bash
echo "[ascend910b] Generating ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e ..."
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
res=$(opc $1 --main_func=three_dim_tensor_matrix_multiplication_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ThreeDimTensorMatrixMultiplicationCustom/build_out/op_kernel/ThreeDimTensorMatrixMultiplicationCustom_ascend910b/bin_param/ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e.json ; then
  echo "$2/ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e.json not generated!"
  exit 1
fi

if ! test -f $2/ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e.o ; then
  echo "$2/ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ThreeDimTensorMatrixMultiplicationCustom_0577b10ad580b932105f76ccf8e6af1e Done"
