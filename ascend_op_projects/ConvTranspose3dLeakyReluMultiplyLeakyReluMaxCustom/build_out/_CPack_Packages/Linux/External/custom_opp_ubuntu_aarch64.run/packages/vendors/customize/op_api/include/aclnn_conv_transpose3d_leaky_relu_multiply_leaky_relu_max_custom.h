
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV_TRANSPOSE3D_LEAKY_RELU_MULTIPLY_LEAKY_RELU_MAX_CUSTOM_H_
#define ACLNN_CONV_TRANSPOSE3D_LEAKY_RELU_MULTIPLY_LEAKY_RELU_MAX_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomGetWorkspaceSize
 * parameters :
 * x : required
 * multiplier : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *multiplier,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
