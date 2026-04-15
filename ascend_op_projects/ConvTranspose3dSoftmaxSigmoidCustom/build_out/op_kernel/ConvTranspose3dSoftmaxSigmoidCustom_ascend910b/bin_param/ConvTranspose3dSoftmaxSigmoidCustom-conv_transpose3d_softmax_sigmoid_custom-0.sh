#!/bin/bash
echo "[ascend910b] Generating ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452 ..."
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
res=$(opc $1 --main_func=conv_transpose3d_softmax_sigmoid_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/ConvTranspose3dSoftmaxSigmoidCustom/build_out/op_kernel/ConvTranspose3dSoftmaxSigmoidCustom_ascend910b/bin_param/ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452.json ; then
  echo "$2/ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452.json not generated!"
  exit 1
fi

if ! test -f $2/ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452.o ; then
  echo "$2/ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating ConvTranspose3dSoftmaxSigmoidCustom_b3a528092d2d8307dbe1117fe7f4e452 Done"
