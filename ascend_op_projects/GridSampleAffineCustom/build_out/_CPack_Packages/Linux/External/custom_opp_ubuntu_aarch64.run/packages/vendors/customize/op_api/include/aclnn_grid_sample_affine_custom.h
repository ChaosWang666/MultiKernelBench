
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRID_SAMPLE_AFFINE_CUSTOM_H_
#define ACLNN_GRID_SAMPLE_AFFINE_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGridSampleAffineCustomGetWorkspaceSize
 * parameters :
 * x : required
 * theta : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampleAffineCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *theta,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGridSampleAffineCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampleAffineCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
