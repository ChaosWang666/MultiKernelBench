
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_BIAS_FOUR_DIM_BROADCAST_CUSTOM_H_
#define ACLNN_ADD_BIAS_FOUR_DIM_BROADCAST_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddBiasFourDimBroadcastCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddBiasFourDimBroadcastCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddBiasFourDimBroadcastCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddBiasFourDimBroadcastCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
