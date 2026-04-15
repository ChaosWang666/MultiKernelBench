
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CUMSUM_EXCLUSIVE_CUSTOM_H_
#define ACLNN_CUMSUM_EXCLUSIVE_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnCumsumExclusiveCustomGetWorkspaceSize
 * parameters :
 * x : required
 * dim : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCumsumExclusiveCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnCumsumExclusiveCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnCumsumExclusiveCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
