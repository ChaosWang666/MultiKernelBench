
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ARGMAX_OVER_ADIMENSION_CUSTOM_H_
#define ACLNN_ARGMAX_OVER_ADIMENSION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnArgmaxOverADimensionCustomGetWorkspaceSize
 * parameters :
 * x : required
 * dim : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnArgmaxOverADimensionCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnArgmaxOverADimensionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnArgmaxOverADimensionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
