#!/bin/bash
echo "[ascend910b] Generating GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467 ..."
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
res=$(opc $1 --main_func=gemm_batch_norm_scaling_softmax_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GemmBatchNormScalingSoftmaxCustom/build_out/op_kernel/GemmBatchNormScalingSoftmaxCustom_ascend910b/bin_param/GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467.json ; then
  echo "$2/GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467.json not generated!"
  exit 1
fi

if ! test -f $2/GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467.o ; then
  echo "$2/GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GemmBatchNormScalingSoftmaxCustom_5058a7829f153184446d57277fa96467 Done"
