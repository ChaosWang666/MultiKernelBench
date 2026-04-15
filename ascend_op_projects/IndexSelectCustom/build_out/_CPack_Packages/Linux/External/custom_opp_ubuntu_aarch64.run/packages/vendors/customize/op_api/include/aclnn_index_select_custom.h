
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INDEX_SELECT_CUSTOM_H_
#define ACLNN_INDEX_SELECT_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnIndexSelectCustomGetWorkspaceSize
 * parameters :
 * x : required
 * indices : required
 * dim : required
 * numRows : required
 * numCols : required
 * numIndices : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIndexSelectCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *indices,
    int64_t dim,
    int64_t numRows,
    int64_t numCols,
    int64_t numIndices,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnIndexSelectCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnIndexSelectCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
