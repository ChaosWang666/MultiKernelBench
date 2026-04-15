
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SGD_CUSTOM_H_
#define ACLNN_SGD_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSgdCustomGetWorkspaceSize
 * parameters :
 * param : required
 * grad : required
 * velocity : required
 * momentum : optional
 * lr : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSgdCustomGetWorkspaceSize(
    const aclTensor *param,
    const aclTensor *grad,
    const aclTensor *velocity,
    double momentum,
    double lr,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSgdCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSgdCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
