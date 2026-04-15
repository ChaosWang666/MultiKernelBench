
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TAKE_ALONG_DIM_CUSTOM_H_
#define ACLNN_TAKE_ALONG_DIM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTakeAlongDimCustomGetWorkspaceSize
 * parameters :
 * x : required
 * idx : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTakeAlongDimCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *idx,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTakeAlongDimCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTakeAlongDimCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
