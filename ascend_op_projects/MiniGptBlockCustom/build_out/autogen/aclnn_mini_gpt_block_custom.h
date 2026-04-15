
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MINI_GPT_BLOCK_CUSTOM_H_
#define ACLNN_MINI_GPT_BLOCK_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMiniGptBlockCustomGetWorkspaceSize
 * parameters :
 * x : required
 * ln1Weight : required
 * ln1Bias : required
 * attnCAttnWeight : required
 * attnCAttnBias : required
 * attnCProjWeight : required
 * attnCProjBias : required
 * ln2Weight : required
 * ln2Bias : required
 * mlpCFcWeight : required
 * mlpCFcBias : required
 * mlpCProjWeight : required
 * mlpCProjBias : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMiniGptBlockCustomGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *ln1Weight,
    const aclTensor *ln1Bias,
    const aclTensor *attnCAttnWeight,
    const aclTensor *attnCAttnBias,
    const aclTensor *attnCProjWeight,
    const aclTensor *attnCProjBias,
    const aclTensor *ln2Weight,
    const aclTensor *ln2Bias,
    const aclTensor *mlpCFcWeight,
    const aclTensor *mlpCFcBias,
    const aclTensor *mlpCProjWeight,
    const aclTensor *mlpCProjBias,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMiniGptBlockCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMiniGptBlockCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
