#!/bin/bash
echo "[ascend910b] Generating EluCustom_6d395e6ee055fdc7337ec77191f16f43 ..."
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
res=$(opc $1 --main_func=elu_custom --input_param=/data/w00936672/MultiKernelBench/ascend_op_projects/EluCustom/build_out/op_kernel/EluCustom_ascend910b/bin_param/EluCustom_6d395e6ee055fdc7337ec77191f16f43_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/EluCustom_6d395e6ee055fdc7337ec77191f16f43.json ; then
  echo "$2/EluCustom_6d395e6ee055fdc7337ec77191f16f43.json not generated!"
  exit 1
fi

if ! test -f $2/EluCustom_6d395e6ee055fdc7337ec77191f16f43.o ; then
  echo "$2/EluCustom_6d395e6ee055fdc7337ec77191f16f43.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating EluCustom_6d395e6ee055fdc7337ec77191f16f43 Done"
