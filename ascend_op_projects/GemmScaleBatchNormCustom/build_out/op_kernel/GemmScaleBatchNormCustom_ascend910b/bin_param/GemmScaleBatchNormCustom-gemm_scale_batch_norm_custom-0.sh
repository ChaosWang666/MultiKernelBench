#!/bin/bash
echo "[ascend910b] Generating GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59 ..."
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
res=$(opc $1 --main_func=gemm_scale_batch_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GemmScaleBatchNormCustom/build_out/op_kernel/GemmScaleBatchNormCustom_ascend910b/bin_param/GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59.json ; then
  echo "$2/GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59.json not generated!"
  exit 1
fi

if ! test -f $2/GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59.o ; then
  echo "$2/GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GemmScaleBatchNormCustom_d50deb866d66ebaf7b7d4b0487e74c59 Done"
