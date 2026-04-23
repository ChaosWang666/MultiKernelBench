
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATMUL_BATCH_NORM_BIAS_ADD_DIVIDE_SWISH_CUSTOM_H_
#define ACLNN_MATMUL_BATCH_NORM_BIAS_ADD_DIVIDE_SWISH_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatmulBatchNormBiasAddDivideSwishCustomGetWorkspaceSize
 * parameters :
 * x : required
 * bias : required
 * divideValue : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulBatchNormBiasAddDivideSwishCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *bias,
    double divideValue,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatmulBatchNormBiasAddDivideSwishCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulBatchNormBiasAddDivideSwishCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
