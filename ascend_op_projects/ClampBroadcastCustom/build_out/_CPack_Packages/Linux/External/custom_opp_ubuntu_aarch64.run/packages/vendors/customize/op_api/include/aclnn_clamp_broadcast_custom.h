
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CLAMP_BROADCAST_CUSTOM_H_
#define ACLNN_CLAMP_BROADCAST_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnClampBroadcastCustomGetWorkspaceSize
 * parameters :
 * x : required
 * minVal : required
 * maxVal : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClampBroadcastCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *minVal,
    const aclTensor *maxVal,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnClampBroadcastCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnClampBroadcastCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
