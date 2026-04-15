#include "kernel_operator.h"

extern "C" __global__ __aicore__ void max_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}