
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MATMUL_ADD_SWISH_TANH_GELU_HARDTANH_CUSTOM_H_
#define ACLNN_MATMUL_ADD_SWISH_TANH_GELU_HARDTANH_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMatmulAddSwishTanhGeluHardtanhCustomGetWorkspaceSize
 * parameters :
 * x : required
 * addValue : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulAddSwishTanhGeluHardtanhCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *addValue,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMatmulAddSwishTanhGeluHardtanhCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMatmulAddSwishTanhGeluHardtanhCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
