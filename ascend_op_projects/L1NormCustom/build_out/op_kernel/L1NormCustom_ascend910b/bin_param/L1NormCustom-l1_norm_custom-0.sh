#!/bin/bash
echo "[ascend910b] Generating L1NormCustom_58346c540424b170944c6993643ab815 ..."
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
res=$(opc $1 --main_func=l1_norm_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/L1NormCustom/build_out/op_kernel/L1NormCustom_ascend910b/bin_param/L1NormCustom_58346c540424b170944c6993643ab815_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/L1NormCustom_58346c540424b170944c6993643ab815.json ; then
  echo "$2/L1NormCustom_58346c540424b170944c6993643ab815.json not generated!"
  exit 1
fi

if ! test -f $2/L1NormCustom_58346c540424b170944c6993643ab815.o ; then
  echo "$2/L1NormCustom_58346c540424b170944c6993643ab815.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating L1NormCustom_58346c540424b170944c6993643ab815 Done"
