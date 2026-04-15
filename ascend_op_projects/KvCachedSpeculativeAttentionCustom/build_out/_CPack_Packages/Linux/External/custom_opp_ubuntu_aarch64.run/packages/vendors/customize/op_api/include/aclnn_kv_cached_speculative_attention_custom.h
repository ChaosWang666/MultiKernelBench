
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_KV_CACHED_SPECULATIVE_ATTENTION_CUSTOM_H_
#define ACLNN_KV_CACHED_SPECULATIVE_ATTENTION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnKvCachedSpeculativeAttentionCustomGetWorkspaceSize
 * parameters :
 * query : required
 * keyCache : required
 * valueCache : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvCachedSpeculativeAttentionCustomGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnKvCachedSpeculativeAttentionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnKvCachedSpeculativeAttentionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
