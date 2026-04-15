
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GRID_SAMPLE_RANDOM_WARP_CUSTOM_H_
#define ACLNN_GRID_SAMPLE_RANDOM_WARP_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGridSampleRandomWarpCustomGetWorkspaceSize
 * parameters :
 * x : required
 * grid : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampleRandomWarpCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *grid,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGridSampleRandomWarpCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGridSampleRandomWarpCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
