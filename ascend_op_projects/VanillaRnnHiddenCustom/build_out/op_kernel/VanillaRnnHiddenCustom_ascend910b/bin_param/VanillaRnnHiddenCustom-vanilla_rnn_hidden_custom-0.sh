#!/bin/bash
echo "[ascend910b] Generating VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07 ..."
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
res=$(opc $1 --main_func=vanilla_rnn_hidden_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/VanillaRnnHiddenCustom/build_out/op_kernel/VanillaRnnHiddenCustom_ascend910b/bin_param/VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07.json ; then
  echo "$2/VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07.json not generated!"
  exit 1
fi

if ! test -f $2/VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07.o ; then
  echo "$2/VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating VanillaRnnHiddenCustom_4ce156c8e5e9b3a359ead3f2ffef6d07 Done"
