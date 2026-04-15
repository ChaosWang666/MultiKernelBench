#!/bin/bash
echo "[ascend910b] Generating RmsNormCustom_2199f396b0b1df533833bdfac79eab6b ..."
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
res=$(opc $1 --main_func=rms_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/RmsNormCustom/build_out/op_kernel/RmsNormCustom_ascend910b/bin_param/RmsNormCustom_2199f396b0b1df533833bdfac79eab6b_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/RmsNormCustom_2199f396b0b1df533833bdfac79eab6b.json ; then
  echo "$2/RmsNormCustom_2199f396b0b1df533833bdfac79eab6b.json not generated!"
  exit 1
fi

if ! test -f $2/RmsNormCustom_2199f396b0b1df533833bdfac79eab6b.o ; then
  echo "$2/RmsNormCustom_2199f396b0b1df533833bdfac79eab6b.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating RmsNormCustom_2199f396b0b1df533833bdfac79eab6b Done"
