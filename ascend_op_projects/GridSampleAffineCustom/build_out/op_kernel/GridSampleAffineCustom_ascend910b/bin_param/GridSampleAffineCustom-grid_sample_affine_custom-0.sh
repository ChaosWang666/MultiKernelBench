#!/bin/bash
echo "[ascend910b] Generating GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7 ..."
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
res=$(opc $1 --main_func=grid_sample_affine_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/GridSampleAffineCustom/build_out/op_kernel/GridSampleAffineCustom_ascend910b/bin_param/GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7.json ; then
  echo "$2/GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7.json not generated!"
  exit 1
fi

if ! test -f $2/GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7.o ; then
  echo "$2/GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating GridSampleAffineCustom_c7529e37bbb60b70a973cbf00b776bc7 Done"
