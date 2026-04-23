
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONVTRANSPOSE2D_SOFTMAX_BIASADD_SCALING_SIGMOID_CUSTOM_H_
#define ACLNN_CONVTRANSPOSE2D_SOFTMAX_BIASADD_SCALING_SIGMOID_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvtranspose2dSoftmaxBiasaddScalingSigmoidCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * scalingFactor : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvtranspose2dSoftmaxBiasaddScalingSigmoidCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    double scalingFactor,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvtranspose2dSoftmaxBiasaddScalingSigmoidCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvtranspose2dSoftmaxBiasaddScalingSigmoidCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
