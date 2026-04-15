
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_THREE_DIM_TENSOR_MATRIX_MULTIPLICATION_CUSTOM_H_
#define ACLNN_THREE_DIM_TENSOR_MATRIX_MULTIPLICATION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnThreeDimTensorMatrixMultiplicationCustomGetWorkspaceSize
 * parameters :
 * a : required
 * b : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnThreeDimTensorMatrixMultiplicationCustomGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnThreeDimTensorMatrixMultiplicationCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnThreeDimTensorMatrixMultiplicationCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
