
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_HINGE_LOSS_CUSTOM_H_
#define ACLNN_HINGE_LOSS_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnHingeLossCustomGetWorkspaceSize
 * parameters :
 * predictions : required
 * targets : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHingeLossCustomGetWorkspaceSize(
    const aclTensor *predictions,
    const aclTensor *targets,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnHingeLossCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnHingeLossCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
