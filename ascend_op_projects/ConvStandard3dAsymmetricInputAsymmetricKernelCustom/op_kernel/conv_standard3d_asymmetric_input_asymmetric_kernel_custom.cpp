#include "kernel_operator.h"

extern "C" __global__ __aicore__ void conv_standard3d_asymmetric_input_asymmetric_kernel_custom(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}