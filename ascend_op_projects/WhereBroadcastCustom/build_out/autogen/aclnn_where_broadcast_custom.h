
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_WHERE_BROADCAST_CUSTOM_H_
#define ACLNN_WHERE_BROADCAST_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnWhereBroadcastCustomGetWorkspaceSize
 * parameters :
 * cond : required
 * x : required
 * y : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnWhereBroadcastCustomGetWorkspaceSize(
    const aclTensor *cond,
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnWhereBroadcastCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnWhereBroadcastCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
