
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMM_GROUP_NORM_SWISH_MULTIPLY_SWISH_CUSTOM_H_
#define ACLNN_GEMM_GROUP_NORM_SWISH_MULTIPLY_SWISH_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmGroupNormSwishMultiplySwishCustomGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * bias : required
 * groupNormWeight : required
 * groupNormBias : required
 * multiplyWeight : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmGroupNormSwishMultiplySwishCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *bias,
    const aclTensor *groupNormWeight,
    const aclTensor *groupNormBias,
    const aclTensor *multiplyWeight,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmGroupNormSwishMultiplySwishCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmGroupNormSwishMultiplySwishCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
