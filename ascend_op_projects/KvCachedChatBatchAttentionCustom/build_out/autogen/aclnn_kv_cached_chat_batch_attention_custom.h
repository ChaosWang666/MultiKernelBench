
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_KV_CACHED_CHAT_BATCH_ATTENTION_CUSTOM_H_
#define ACLNN_KV_CACHED_CHAT_BATCH_ATTENTION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnKvCachedChatBatchAttentionCustomGetWorkspaceSize
 * parameters :
 * q : required
 * kCache : required
 * vCache : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvCachedChatBatchAttentionCustomGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *kCache,
    const aclTensor *vCache,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnKvCachedChatBatchAttentionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvCachedChatBatchAttentionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
