#!/bin/bash
echo "[ascend910b] Generating HuberLossCustom_0868b64361c6930ebf12bad2a178cceb ..."
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
res=$(opc $1 --main_func=huber_loss_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/HuberLossCustom/build_out/op_kernel/HuberLossCustom_ascend910b/bin_param/HuberLossCustom_0868b64361c6930ebf12bad2a178cceb_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/HuberLossCustom_0868b64361c6930ebf12bad2a178cceb.json ; then
  echo "$2/HuberLossCustom_0868b64361c6930ebf12bad2a178cceb.json not generated!"
  exit 1
fi

if ! test -f $2/HuberLossCustom_0868b64361c6930ebf12bad2a178cceb.o ; then
  echo "$2/HuberLossCustom_0868b64361c6930ebf12bad2a178cceb.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating HuberLossCustom_0868b64361c6930ebf12bad2a178cceb Done"
