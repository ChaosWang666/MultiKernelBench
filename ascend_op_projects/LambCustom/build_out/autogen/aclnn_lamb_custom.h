
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LAMB_CUSTOM_H_
#define ACLNN_LAMB_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLambCustomGetWorkspaceSize
 * parameters :
 * param : required
 * m : required
 * v : required
 * lr : required
 * eps : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLambCustomGetWorkspaceSize(
    const aclTensor *param,
    const aclTensor *m,
    const aclTensor *v,
    double lr,
    double eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLambCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLambCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
