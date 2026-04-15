
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADAM_CUSTOM_H_
#define ACLNN_ADAM_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAdamCustomGetWorkspaceSize
 * parameters :
 * param : required
 * grad : required
 * m : required
 * v : required
 * beta1 : required
 * beta2 : required
 * lr : required
 * eps : required
 * step : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAdamCustomGetWorkspaceSize(
    const aclTensor *param,
    const aclTensor *grad,
    const aclTensor *m,
    const aclTensor *v,
    double beta1,
    double beta2,
    double lr,
    double eps,
    int64_t step,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAdamCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAdamCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
