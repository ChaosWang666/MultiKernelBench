
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMM_SCALE_BATCH_NORM_CUSTOM_H_
#define ACLNN_GEMM_SCALE_BATCH_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmScaleBatchNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * scale : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmScaleBatchNormCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *scale,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmScaleBatchNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmScaleBatchNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
