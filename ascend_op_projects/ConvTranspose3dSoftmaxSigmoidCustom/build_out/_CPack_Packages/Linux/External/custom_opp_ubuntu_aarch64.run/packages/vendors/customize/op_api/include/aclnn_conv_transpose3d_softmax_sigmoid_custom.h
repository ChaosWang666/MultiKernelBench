
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV_TRANSPOSE3D_SOFTMAX_SIGMOID_CUSTOM_H_
#define ACLNN_CONV_TRANSPOSE3D_SOFTMAX_SIGMOID_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvTranspose3dSoftmaxSigmoidCustomGetWorkspaceSize
 * parameters :
 * x : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dSoftmaxSigmoidCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvTranspose3dSoftmaxSigmoidCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dSoftmaxSigmoidCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
