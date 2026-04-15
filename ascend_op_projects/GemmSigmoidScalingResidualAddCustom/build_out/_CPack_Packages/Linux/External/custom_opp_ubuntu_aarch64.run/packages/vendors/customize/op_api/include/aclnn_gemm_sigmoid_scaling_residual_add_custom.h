
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMM_SIGMOID_SCALING_RESIDUAL_ADD_CUSTOM_H_
#define ACLNN_GEMM_SIGMOID_SCALING_RESIDUAL_ADD_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmSigmoidScalingResidualAddCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * scalingFactor : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmSigmoidScalingResidualAddCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    double scalingFactor,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmSigmoidScalingResidualAddCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmSigmoidScalingResidualAddCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
