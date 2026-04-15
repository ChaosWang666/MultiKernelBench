#ifndef __ADD_BIAS_BROADCAST_CUSTOM__KERNEL_FUN_H__
#define __ADD_BIAS_BROADCAST_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#include "/mnt/workspace/claude_code/MultiKernelBench/ascend_op_projects/AddBiasBroadcastCustom/op_kernel/add_bias_broadcast_custom.cpp"
#include "kernel_utils.h"
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#define TEMPLATE_PARAMS -1
#define TEMPLATE_PARAMS_LEN 0

#if TILING_KEY_VAR == 0UL && (defined(__DAV_VEC__) && __NPU_ARCH__ == 2201)
    __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {0};
#else
    extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;
#endif

__aicore__ inline void ascendc_auto_gen_add_bias_broadcast_custom_kernel(GM_ADDR x_in__, GM_ADDR bias_in__, GM_ADDR z_out_, GM_ADDR workspace, GM_ADDR tiling) {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    workspace += 78643200;
    #endif
    AscendC::SetSysWorkspaceForce(workspace);
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t ASCENDC_DUMP_SIZE = 123;
    AscendC::InitDump(false, ASCENDC_DUMP_SIZE);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));
#endif
    #endif
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_MC2_CTX));
#endif
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
#if defined(TEMPLATE_PARAMS_LEN) && TEMPLATE_PARAMS_LEN != 0
    add_bias_broadcast_custom<TEMPLATE_PARAMS>(x_in__, bias_in__, z_out_, usrWorkspace, tiling);
#else
    add_bias_broadcast_custom(x_in__, bias_in__, z_out_, usrWorkspace, tiling);
#endif
}

extern "C" __global__ [aicore] void auto_gen_add_bias_broadcast_custom_kernel(GM_ADDR x_in__, GM_ADDR bias_in__, GM_ADDR z_out_, GM_ADDR workspace, GM_ADDR tiling) {
    ascendc_auto_gen_add_bias_broadcast_custom_kernel(x_in__, bias_in__, z_out_, workspace, tiling);
}

// generate dfx section for tiling_key:0
struct AscendCInfoMetaDFX {
    BaseTlv head;
    uint8_t value[92];
};




#if TILING_KEY_VAR == 0UL
static const struct FunLevelKType AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684_0_kernel_type_section __attribute__ ((used, section (".ascend.meta.AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684_0"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_AIV} };
static const struct AscendCInfoMetaDFX AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684_0_dfx_section __attribute__ ((used, section (".ascend.meta.AddBiasBroadcastCustom_49a58dfb8b62a3acf2ae322c61d45684_0"))) = {{4, 92}, { 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 3, 255, 255, 255, 255, 255, 255, 255, 255, 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0, 4, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 7, 0, 0, 0, 0, 0, 0, 0, 16, } };
#endif
#endif
