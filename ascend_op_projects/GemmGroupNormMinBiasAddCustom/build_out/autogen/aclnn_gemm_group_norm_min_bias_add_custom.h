
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMM_GROUP_NORM_MIN_BIAS_ADD_CUSTOM_H_
#define ACLNN_GEMM_GROUP_NORM_MIN_BIAS_ADD_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmGroupNormMinBiasAddCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmGroupNormMinBiasAddCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmGroupNormMinBiasAddCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmGroupNormMinBiasAddCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
