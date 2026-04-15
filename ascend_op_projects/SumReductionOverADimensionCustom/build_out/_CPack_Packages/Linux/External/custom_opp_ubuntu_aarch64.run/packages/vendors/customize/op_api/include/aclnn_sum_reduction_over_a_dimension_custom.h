
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SUM_REDUCTION_OVER_ADIMENSION_CUSTOM_H_
#define ACLNN_SUM_REDUCTION_OVER_ADIMENSION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSumReductionOverADimensionCustomGetWorkspaceSize
 * parameters :
 * x : required
 * dim : required
 * shape0 : required
 * shape1 : required
 * shape2 : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSumReductionOverADimensionCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t dim,
    int64_t shape0,
    int64_t shape1,
    int64_t shape2,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSumReductionOverADimensionCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSumReductionOverADimensionCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
