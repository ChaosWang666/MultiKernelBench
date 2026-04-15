
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RMS_NORM_CUSTOM_H_
#define ACLNN_RMS_NORM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRmsNormCustomGetWorkspaceSize
 * parameters :
 * x : required
 * numFeatures : required
 * eps : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t numFeatures,
    double eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRmsNormCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRmsNormCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
