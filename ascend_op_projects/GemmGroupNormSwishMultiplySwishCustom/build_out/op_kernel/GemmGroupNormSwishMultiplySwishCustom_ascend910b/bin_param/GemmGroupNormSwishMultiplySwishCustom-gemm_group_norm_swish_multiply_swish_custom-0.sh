#!/bin/bash
echo "[ascend910b] Generating GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d ..."
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
res=$(opc $1 --main_func=gemm_group_norm_swish_multiply_swish_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GemmGroupNormSwishMultiplySwishCustom/build_out/op_kernel/GemmGroupNormSwishMultiplySwishCustom_ascend910b/bin_param/GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d.json ; then
  echo "$2/GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d.json not generated!"
  exit 1
fi

if ! test -f $2/GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d.o ; then
  echo "$2/GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GemmGroupNormSwishMultiplySwishCustom_52b705a57b7ed884492a23b44ecee03d Done"
