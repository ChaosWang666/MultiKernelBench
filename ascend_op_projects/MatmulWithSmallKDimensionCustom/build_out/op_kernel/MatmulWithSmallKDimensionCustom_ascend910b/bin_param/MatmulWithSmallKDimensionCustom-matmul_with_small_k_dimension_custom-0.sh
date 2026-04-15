#!/bin/bash
echo "[ascend910b] Generating MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7 ..."
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
res=$(opc $1 --main_func=matmul_with_small_k_dimension_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MatmulWithSmallKDimensionCustom/build_out/op_kernel/MatmulWithSmallKDimensionCustom_ascend910b/bin_param/MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7.json ; then
  echo "$2/MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7.json not generated!"
  exit 1
fi

if ! test -f $2/MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7.o ; then
  echo "$2/MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MatmulWithSmallKDimensionCustom_6d15d1a02f99029ba809958af06741d7 Done"
