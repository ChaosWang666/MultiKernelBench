
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_BATCHED_MATRIX_MULTIPLICATION_CUSTOM_H_
#define ACLNN_BATCHED_MATRIX_MULTIPLICATION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnBatchedMatrixMultiplicationCustomGetWorkspaceSize
 * parameters :
 * a : required
 * b : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchedMatrixMultiplicationCustomGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnBatchedMatrixMultiplicationCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnBatchedMatrixMultiplicationCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
