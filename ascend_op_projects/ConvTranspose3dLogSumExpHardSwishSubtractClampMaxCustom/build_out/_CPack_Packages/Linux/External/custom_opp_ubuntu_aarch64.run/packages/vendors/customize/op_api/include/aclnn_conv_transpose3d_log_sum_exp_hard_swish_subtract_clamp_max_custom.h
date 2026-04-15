
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV_TRANSPOSE3D_LOG_SUM_EXP_HARD_SWISH_SUBTRACT_CLAMP_MAX_CUSTOM_H_
#define ACLNN_CONV_TRANSPOSE3D_LOG_SUM_EXP_HARD_SWISH_SUBTRACT_CLAMP_MAX_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dLogSumExpHardSwishSubtractClampMaxCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
