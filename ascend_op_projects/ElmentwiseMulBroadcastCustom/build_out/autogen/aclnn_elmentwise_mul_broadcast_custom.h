
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ELMENTWISE_MUL_BROADCAST_CUSTOM_H_
#define ACLNN_ELMENTWISE_MUL_BROADCAST_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnElmentwiseMulBroadcastCustomGetWorkspaceSize
 * parameters :
 * x : required
 * y : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnElmentwiseMulBroadcastCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnElmentwiseMulBroadcastCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnElmentwiseMulBroadcastCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
