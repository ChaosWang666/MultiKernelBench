
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATMUL_SCALE_RESIDUAL_ADD_CLAMP_LOG_SUM_EXP_MISH_CUSTOM_H_
#define ACLNN_MATMUL_SCALE_RESIDUAL_ADD_CLAMP_LOG_SUM_EXP_MISH_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatmulScaleResidualAddClampLogSumExpMishCustomGetWorkspaceSize
 * parameters :
 * x : required
 * scaleFactor : required
 * clampMin : required
 * clampMax : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulScaleResidualAddClampLogSumExpMishCustomGetWorkspaceSize(
    const aclTensor *x,
    double scaleFactor,
    double clampMin,
    double clampMax,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatmulScaleResidualAddClampLogSumExpMishCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulScaleResidualAddClampLogSumExpMishCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
