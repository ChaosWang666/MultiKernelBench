
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MASKED_FILL_CUSTOM_H_
#define ACLNN_MASKED_FILL_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMaskedFillCustomGetWorkspaceSize
 * parameters :
 * x : required
 * mask : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedFillCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *mask,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMaskedFillCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedFillCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
