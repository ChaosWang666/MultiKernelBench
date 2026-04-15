#!/bin/bash
echo "[ascend910b] Generating MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b ..."
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
res=$(opc $1 --main_func=mamba_return_final_state_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/MambaReturnFinalStateCustom/build_out/op_kernel/MambaReturnFinalStateCustom_ascend910b/bin_param/MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b.json ; then
  echo "$2/MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b.json not generated!"
  exit 1
fi

if ! test -f $2/MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b.o ; then
  echo "$2/MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating MambaReturnFinalStateCustom_a14e196c2781c9dc6154959db78e579b Done"
