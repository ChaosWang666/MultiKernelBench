
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MULTI_QUERY_ATTENTION_CUSTOM_H_
#define ACLNN_MULTI_QUERY_ATTENTION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMultiQueryAttentionCustomGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * batchSize : required
 * seqLen : required
 * numHeads : required
 * headDim : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiQueryAttentionCustomGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    int64_t batchSize,
    int64_t seqLen,
    int64_t numHeads,
    int64_t headDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMultiQueryAttentionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMultiQueryAttentionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
