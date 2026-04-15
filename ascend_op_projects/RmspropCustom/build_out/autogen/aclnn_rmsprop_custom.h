
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMSPROP_CUSTOM_H_
#define ACLNN_RMSPROP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmspropCustomGetWorkspaceSize
 * parameters :
 * param : required
 * grad : required
 * v : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmspropCustomGetWorkspaceSize(
    const aclTensor *param,
    const aclTensor *grad,
    const aclTensor *v,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmspropCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmspropCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
