#!/bin/bash
echo "[ascend910b] Generating CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108 ..."
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
res=$(opc $1 --main_func=cosine_similarity_loss_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/CosineSimilarityLossCustom/build_out/op_kernel/CosineSimilarityLossCustom_ascend910b/bin_param/CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108.json ; then
  echo "$2/CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108.json not generated!"
  exit 1
fi

if ! test -f $2/CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108.o ; then
  echo "$2/CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating CosineSimilarityLossCustom_e7ed7299463a98cfdc8efbd112bde108 Done"
