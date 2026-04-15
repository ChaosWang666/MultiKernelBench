
#include "kernel_operator.h"

// grid_sample bilinear, align_corners=False
// x: [N, C, inH, inW], grid: [N, outH, outW, 2], y: [N, C, outH, outW]
// Each AI core processes one batch element (BLOCK_DIM == N)

extern "C" __global__ __aicore__ void grid_sample_random_warp_custom(GM_ADDR x, GM_ADDR grid, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);

    uint32_t batchSize = t.batchSize;
    uint32_t C = t.channels;
    uint32_t inH = t.inH;
    uint32_t inW = t.inW;
    uint32_t outH = t.outH;
    uint32_t outW = t.outW;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= batchSize) return;

    // Pointers for this batch
    __gm__ float* xPtr = ((__gm__ float*)x) + blockIdx * C * inH * inW;
    __gm__ float* gridPtr = ((__gm__ float*)grid) + blockIdx * outH * outW * 2;
    __gm__ float* yPtr = ((__gm__ float*)y) + blockIdx * C * outH * outW;

    // Process each output pixel
    for (uint32_t oh = 0; oh < outH; oh++) {
        for (uint32_t ow = 0; ow < outW; ow++) {
            uint32_t gridIdx = (oh * outW + ow) * 2;

            // Read grid values
            float gx = *(gridPtr + gridIdx);
            float gy = *(gridPtr + gridIdx + 1);

            // Unnormalize: align_corners=False
            // ix = ((gx + 1) / 2) * inW - 0.5
            // iy = ((gy + 1) / 2) * inH - 0.5
            float ix = ((gx + 1.0f) * 0.5f) * (float)inW - 0.5f;
            float iy = ((gy + 1.0f) * 0.5f) * (float)inH - 0.5f;

            int32_t ix0 = (int32_t)ix;
            if (ix < 0) ix0 = ix0 - 1;
            if ((float)ix0 == ix && ix >= 0) ix0 = ix0; // floor
            else if (ix < 0) {
                ix0 = (int32_t)(ix) - (ix != (int32_t)(ix) ? 1 : 0);
            }
            // Proper floor
            ix0 = (ix >= 0) ? (int32_t)ix : ((int32_t)ix - ((float)(int32_t)ix != ix ? 1 : 0));
            int32_t iy0 = (iy >= 0) ? (int32_t)iy : ((int32_t)iy - ((float)(int32_t)iy != iy ? 1 : 0));

            int32_t ix1 = ix0 + 1;
            int32_t iy1 = iy0 + 1;

            float dx = ix - (float)ix0;
            float dy = iy - (float)iy0;

            float w00 = (1.0f - dx) * (1.0f - dy);
            float w01 = dx * (1.0f - dy);
            float w10 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            for (uint32_t c = 0; c < C; c++) {
                __gm__ float* xc = xPtr + c * inH * inW;
                float val = 0.0f;

                // Top-left
                if (iy0 >= 0 && iy0 < (int32_t)inH && ix0 >= 0 && ix0 < (int32_t)inW) {
                    val += w00 * *(xc + iy0 * (int32_t)inW + ix0);
                }
                // Top-right
                if (iy0 >= 0 && iy0 < (int32_t)inH && ix1 >= 0 && ix1 < (int32_t)inW) {
                    val += w01 * *(xc + iy0 * (int32_t)inW + ix1);
                }
                // Bottom-left
                if (iy1 >= 0 && iy1 < (int32_t)inH && ix0 >= 0 && ix0 < (int32_t)inW) {
                    val += w10 * *(xc + iy1 * (int32_t)inW + ix0);
                }
                // Bottom-right
                if (iy1 >= 0 && iy1 < (int32_t)inH && ix1 >= 0 && ix1 < (int32_t)inW) {
                    val += w11 * *(xc + iy1 * (int32_t)inW + ix1);
                }

                *(yPtr + c * outH * outW + oh * outW + ow) = val;
            }
        }
    }
}
