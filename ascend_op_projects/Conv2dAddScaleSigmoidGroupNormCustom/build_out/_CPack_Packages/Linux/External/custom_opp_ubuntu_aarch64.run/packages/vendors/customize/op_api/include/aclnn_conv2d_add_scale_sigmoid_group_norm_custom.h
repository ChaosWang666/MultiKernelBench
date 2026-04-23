
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV2D_ADD_SCALE_SIGMOID_GROUP_NORM_CUSTOM_H_
#define ACLNN_CONV2D_ADD_SCALE_SIGMOID_GROUP_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv2dAddScaleSigmoidGroupNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * scale : required
 * gamma : required
 * beta : required
 * numGroups : required
 * eps : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dAddScaleSigmoidGroupNormCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    const aclTensor *scale,
    const aclTensor *gamma,
    const aclTensor *beta,
    int64_t numGroups,
    double eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv2dAddScaleSigmoidGroupNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv2dAddScaleSigmoidGroupNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
