
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BATCH_NORM_CUSTOM_H_
#define ACLNN_BATCH_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBatchNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * runningMean : required
 * runningVar : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchNormCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *runningMean,
    const aclTensor *runningVar,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBatchNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
