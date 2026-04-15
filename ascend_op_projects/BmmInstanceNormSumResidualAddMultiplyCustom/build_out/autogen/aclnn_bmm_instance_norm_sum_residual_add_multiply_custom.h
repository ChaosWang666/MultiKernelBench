
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BMM_INSTANCE_NORM_SUM_RESIDUAL_ADD_MULTIPLY_CUSTOM_H_
#define ACLNN_BMM_INSTANCE_NORM_SUM_RESIDUAL_ADD_MULTIPLY_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBmmInstanceNormSumResidualAddMultiplyCustomGetWorkspaceSize
 * parameters :
 * x : required
 * y : required
 * weight : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBmmInstanceNormSumResidualAddMultiplyCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *y,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBmmInstanceNormSumResidualAddMultiplyCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBmmInstanceNormSumResidualAddMultiplyCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
