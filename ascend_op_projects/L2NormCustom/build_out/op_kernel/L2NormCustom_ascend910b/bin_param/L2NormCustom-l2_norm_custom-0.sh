#!/bin/bash
echo "[ascend910b] Generating L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f ..."
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
res=$(opc $1 --main_func=l2_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/L2NormCustom/build_out/op_kernel/L2NormCustom_ascend910b/bin_param/L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f.json ; then
  echo "$2/L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f.json not generated!"
  exit 1
fi

if ! test -f $2/L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f.o ; then
  echo "$2/L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating L2NormCustom_2d8332b00c450b1ab98c6ddd76ad2a2f Done"
