#!/bin/bash
echo "[ascend910b] Generating FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a ..."
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
res=$(opc $1 --main_func=frobenius_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/FrobeniusNormCustom/build_out/op_kernel/FrobeniusNormCustom_ascend910b/bin_param/FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a.json ; then
  echo "$2/FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a.json not generated!"
  exit 1
fi

if ! test -f $2/FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a.o ; then
  echo "$2/FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating FrobeniusNormCustom_776161a0835ac593396310ce3b6a736a Done"
