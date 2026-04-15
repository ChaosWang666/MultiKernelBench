#!/bin/bash
echo "[ascend910b] Generating LambCustom_f7a97529922d2e9e6c5ce329402190da ..."
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
res=$(opc $1 --main_func=lamb_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/LambCustom/build_out/op_kernel/LambCustom_ascend910b/bin_param/LambCustom_f7a97529922d2e9e6c5ce329402190da_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/LambCustom_f7a97529922d2e9e6c5ce329402190da.json ; then
  echo "$2/LambCustom_f7a97529922d2e9e6c5ce329402190da.json not generated!"
  exit 1
fi

if ! test -f $2/LambCustom_f7a97529922d2e9e6c5ce329402190da.o ; then
  echo "$2/LambCustom_f7a97529922d2e9e6c5ce329402190da.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating LambCustom_f7a97529922d2e9e6c5ce329402190da Done"
