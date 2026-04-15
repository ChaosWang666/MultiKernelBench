#!/bin/bash
echo "[ascend910b] Generating MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec ..."
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
res=$(opc $1 --main_func=matmul_swish_sum_group_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MatmulSwishSumGroupNormCustom/build_out/op_kernel/MatmulSwishSumGroupNormCustom_ascend910b/bin_param/MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec.json ; then
  echo "$2/MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec.json not generated!"
  exit 1
fi

if ! test -f $2/MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec.o ; then
  echo "$2/MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MatmulSwishSumGroupNormCustom_d43b87faa9991987ea8a9aa2a7be3dec Done"
