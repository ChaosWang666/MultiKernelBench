
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_PRODUCT_REDUCTION_OVER_ADIMENSION_CUSTOM_H_
#define ACLNN_PRODUCT_REDUCTION_OVER_ADIMENSION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnProductReductionOverADimensionCustomGetWorkspaceSize
 * parameters :
 * x : required
 * dim : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnProductReductionOverADimensionCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnProductReductionOverADimensionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnProductReductionOverADimensionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
