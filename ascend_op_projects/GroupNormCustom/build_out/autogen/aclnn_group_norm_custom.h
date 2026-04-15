
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GROUP_NORM_CUSTOM_H_
#define ACLNN_GROUP_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGroupNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * beta : required
 * numGroups : required
 * numChannels : required
 * numHw : required
 * batchSize : required
 * eps : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t numGroups,
    int64_t numChannels,
    int64_t numHw,
    int64_t batchSize,
    double eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGroupNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGroupNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
