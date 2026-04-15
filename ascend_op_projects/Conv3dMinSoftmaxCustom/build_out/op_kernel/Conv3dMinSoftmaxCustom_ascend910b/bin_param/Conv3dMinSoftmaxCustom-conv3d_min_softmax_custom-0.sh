#!/bin/bash
echo "[ascend910b] Generating Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b ..."
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
res=$(opc $1 --main_func=conv3d_min_softmax_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/Conv3dMinSoftmaxCustom/build_out/op_kernel/Conv3dMinSoftmaxCustom_ascend910b/bin_param/Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b.json ; then
  echo "$2/Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b.json not generated!"
  exit 1
fi

if ! test -f $2/Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b.o ; then
  echo "$2/Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating Conv3dMinSoftmaxCustom_3e4e113b9d0446c54a2dfc7f3c30131b Done"
