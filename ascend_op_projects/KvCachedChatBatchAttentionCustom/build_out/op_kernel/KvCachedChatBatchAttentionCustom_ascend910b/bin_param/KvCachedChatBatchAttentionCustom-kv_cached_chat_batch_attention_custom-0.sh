#!/bin/bash
echo "[ascend910b] Generating KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc ..."
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
res=$(opc $1 --main_func=kv_cached_chat_batch_attention_custom --input_param=/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/KvCachedChatBatchAttentionCustom/build_out/op_kernel/KvCachedChatBatchAttentionCustom_ascend910b/bin_param/KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc_param.json --soc_version=Ascend910B1                 --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )

echo "${res}"

if ! test -f $2/KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc.json ; then
  echo "$2/KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc.json not generated!"
  exit 1
fi

if ! test -f $2/KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc.o ; then
  echo "$2/KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc.o not generated!"
  exit 1
fi
echo "[ascend910b] Generating KvCachedChatBatchAttentionCustom_0ac88062b3c2d075fede7f67c838a5cc Done"
