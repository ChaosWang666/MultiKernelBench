
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRU_CUSTOM_H_
#define ACLNN_GRU_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGruCustomGetWorkspaceSize
 * parameters :
 * x : required
 * hx : required
 * wIh : required
 * wHh : required
 * bIh : required
 * bHh : required
 * outputOut : required
 * hyOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGruCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *hx,
    const aclTensor *wIh,
    const aclTensor *wHh,
    const aclTensor *bIh,
    const aclTensor *bHh,
    const aclTensor *outputOut,
    const aclTensor *hyOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGruCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGruCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
