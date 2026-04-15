
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_GEMM_BATCH_NORM_GELU_GROUP_NORM_MEAN_RELU_CUSTOM_H_
#define ACLNN_GEMM_BATCH_NORM_GELU_GROUP_NORM_MEAN_RELU_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGemmBatchNormGeluGroupNormMeanReluCustomGetWorkspaceSize
 * parameters :
 * x : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmBatchNormGeluGroupNormMeanReluCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnGemmBatchNormGeluGroupNormMeanReluCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGemmBatchNormGeluGroupNormMeanReluCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
