#!/bin/bash
echo "[ascend910b] Generating DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d ..."
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
res=$(opc $1 --main_func=deep_narrow_mlp_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/DeepNarrowMlpCustom/build_out/op_kernel/DeepNarrowMlpCustom_ascend910b/bin_param/DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d.json ; then
  echo "$2/DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d.json not generated!"
  exit 1
fi

if ! test -f $2/DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d.o ; then
  echo "$2/DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating DeepNarrowMlpCustom_0aad02854d337af47b75fe9327998d9d Done"
