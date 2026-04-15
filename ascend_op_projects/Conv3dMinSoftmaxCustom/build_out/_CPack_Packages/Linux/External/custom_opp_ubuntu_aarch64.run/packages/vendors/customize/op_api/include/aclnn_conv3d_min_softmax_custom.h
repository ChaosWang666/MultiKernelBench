
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CONV3D_MIN_SOFTMAX_CUSTOM_H_
#define ACLNN_CONV3D_MIN_SOFTMAX_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnConv3dMinSoftmaxCustomGetWorkspaceSize
 * parameters :
 * x : required
 * dimD : required
 * channels : required
 * height : required
 * width : required
 * batchSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv3dMinSoftmaxCustomGetWorkspaceSize(
    const aclTensor *x,
    int64_t dimD,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t batchSize,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnConv3dMinSoftmaxCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnConv3dMinSoftmaxCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
