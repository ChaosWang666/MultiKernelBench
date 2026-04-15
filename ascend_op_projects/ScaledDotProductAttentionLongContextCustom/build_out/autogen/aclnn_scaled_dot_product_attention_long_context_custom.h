
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCALED_DOT_PRODUCT_ATTENTION_LONG_CONTEXT_CUSTOM_H_
#define ACLNN_SCALED_DOT_PRODUCT_ATTENTION_LONG_CONTEXT_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScaledDotProductAttentionLongContextCustomGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScaledDotProductAttentionLongContextCustomGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScaledDotProductAttentionLongContextCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScaledDotProductAttentionLongContextCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
