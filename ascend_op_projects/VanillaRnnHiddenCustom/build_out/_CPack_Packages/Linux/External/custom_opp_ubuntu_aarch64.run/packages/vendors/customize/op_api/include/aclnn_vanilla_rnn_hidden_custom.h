
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_VANILLA_RNN_HIDDEN_CUSTOM_H_
#define ACLNN_VANILLA_RNN_HIDDEN_CUSTOM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnVanillaRnnHiddenCustomGetWorkspaceSize
 * parameters :
 * input : required
 * hidden : required
 * weightIh : required
 * biasIh : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVanillaRnnHiddenCustomGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *hidden,
    const aclTensor *weightIh,
    const aclTensor *biasIh,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnVanillaRnnHiddenCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnVanillaRnnHiddenCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
