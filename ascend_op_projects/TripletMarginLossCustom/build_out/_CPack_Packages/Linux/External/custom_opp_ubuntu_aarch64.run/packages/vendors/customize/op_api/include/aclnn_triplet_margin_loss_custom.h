
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TRIPLET_MARGIN_LOSS_CUSTOM_H_
#define ACLNN_TRIPLET_MARGIN_LOSS_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTripletMarginLossCustomGetWorkspaceSize
 * parameters :
 * anchor : required
 * positive : required
 * negative : required
 * margin : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTripletMarginLossCustomGetWorkspaceSize(
    const aclTensor *anchor,
    const aclTensor *positive,
    const aclTensor *negative,
    double margin,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTripletMarginLossCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTripletMarginLossCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
