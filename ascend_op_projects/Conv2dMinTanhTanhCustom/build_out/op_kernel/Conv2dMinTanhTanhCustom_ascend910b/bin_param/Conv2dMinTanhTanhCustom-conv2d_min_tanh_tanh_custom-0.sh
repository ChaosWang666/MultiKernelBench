#!/bin/bash
echo "[ascend910b] Generating Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a ..."
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
res=$(opc $1 --main_func=conv2d_min_tanh_tanh_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/Conv2dMinTanhTanhCustom/build_out/op_kernel/Conv2dMinTanhTanhCustom_ascend910b/bin_param/Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a.json ; then
  echo "$2/Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a.json not generated!"
  exit 1
fi

if ! test -f $2/Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a.o ; then
  echo "$2/Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating Conv2dMinTanhTanhCustom_00493df0ffa58b728f3065aa932ae77a Done"
