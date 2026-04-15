
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MAMBA_RETURN_YCUSTOM_H_
#define ACLNN_MAMBA_RETURN_YCUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMambaReturnYCustomGetWorkspaceSize
 * parameters :
 * x : required
 * a : required
 * b : required
 * c : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMambaReturnYCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *c,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMambaReturnYCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMambaReturnYCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
