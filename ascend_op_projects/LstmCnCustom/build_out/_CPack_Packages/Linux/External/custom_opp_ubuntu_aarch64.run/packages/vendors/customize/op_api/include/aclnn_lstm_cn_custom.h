
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LSTM_CN_CUSTOM_H_
#define ACLNN_LSTM_CN_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLstmCnCustomGetWorkspaceSize
 * parameters :
 * x : required
 * h0 : required
 * c0 : required
 * hnOut : required
 * cnOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLstmCnCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *h0,
    const aclTensor *c0,
    const aclTensor *hnOut,
    const aclTensor *cnOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLstmCnCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLstmCnCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
