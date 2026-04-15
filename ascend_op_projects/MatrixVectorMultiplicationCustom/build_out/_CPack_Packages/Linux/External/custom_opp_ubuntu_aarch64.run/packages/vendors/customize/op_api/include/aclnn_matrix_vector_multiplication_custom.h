
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATRIX_VECTOR_MULTIPLICATION_CUSTOM_H_
#define ACLNN_MATRIX_VECTOR_MULTIPLICATION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatrixVectorMultiplicationCustomGetWorkspaceSize
 * parameters :
 * a : required
 * b : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatrixVectorMultiplicationCustomGetWorkspaceSize(
    const aclTensor *a,
    const aclTensor *b,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatrixVectorMultiplicationCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatrixVectorMultiplicationCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
