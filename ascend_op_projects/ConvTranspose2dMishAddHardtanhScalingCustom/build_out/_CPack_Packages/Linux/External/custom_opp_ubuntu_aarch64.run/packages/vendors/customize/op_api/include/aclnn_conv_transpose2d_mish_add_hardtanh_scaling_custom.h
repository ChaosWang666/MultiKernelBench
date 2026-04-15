
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV_TRANSPOSE2D_MISH_ADD_HARDTANH_SCALING_CUSTOM_H_
#define ACLNN_CONV_TRANSPOSE2D_MISH_ADD_HARDTANH_SCALING_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvTranspose2dMishAddHardtanhScalingCustomGetWorkspaceSize
 * parameters :
 * x : required
 * addValue : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose2dMishAddHardtanhScalingCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *addValue,
    const aclTensor *scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvTranspose2dMishAddHardtanhScalingCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose2dMishAddHardtanhScalingCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
