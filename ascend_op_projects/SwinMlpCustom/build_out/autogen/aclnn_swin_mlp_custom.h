
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIN_MLP_CUSTOM_H_
#define ACLNN_SWIN_MLP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwinMlpCustomGetWorkspaceSize
 * parameters :
 * x : required
 * fc1Weight : required
 * fc1Bias : required
 * fc2Weight : required
 * fc2Bias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinMlpCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *fc1Weight,
    const aclTensor *fc1Bias,
    const aclTensor *fc2Weight,
    const aclTensor *fc2Bias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwinMlpCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwinMlpCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
