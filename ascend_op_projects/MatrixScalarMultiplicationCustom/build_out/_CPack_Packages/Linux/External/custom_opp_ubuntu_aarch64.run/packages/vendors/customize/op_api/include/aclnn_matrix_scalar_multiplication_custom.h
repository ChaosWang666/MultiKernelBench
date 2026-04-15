
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATRIX_SCALAR_MULTIPLICATION_CUSTOM_H_
#define ACLNN_MATRIX_SCALAR_MULTIPLICATION_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatrixScalarMultiplicationCustomGetWorkspaceSize
 * parameters :
 * x : required
 * s : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatrixScalarMultiplicationCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *s,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatrixScalarMultiplicationCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatrixScalarMultiplicationCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
