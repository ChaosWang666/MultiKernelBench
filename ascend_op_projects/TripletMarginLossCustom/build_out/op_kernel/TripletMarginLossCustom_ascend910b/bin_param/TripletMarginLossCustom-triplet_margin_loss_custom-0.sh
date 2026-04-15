#!/bin/bash
echo "[ascend910b] Generating TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a ..."
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
res=$(opc $1 --main_func=triplet_margin_loss_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/TripletMarginLossCustom/build_out/op_kernel/TripletMarginLossCustom_ascend910b/bin_param/TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a.json ; then
  echo "$2/TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a.json not generated!"
  exit 1
fi

if ! test -f $2/TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a.o ; then
  echo "$2/TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating TripletMarginLossCustom_525b951a11a46727efb8afa9c0473d2a Done"
