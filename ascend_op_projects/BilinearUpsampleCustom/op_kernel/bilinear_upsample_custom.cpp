
#include "kernel_operator.h"

// Bilinear upsample with scale_factor=2, align_corners=False
// Input: [N, C, H, W], Output: [N, C, 2H, 2W]
// For align_corners=False: src_coord = (dst_coord + 0.5) / scale - 0.5

extern "C" __global__ __aicore__ void bilinear_upsample_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t inputH = tiling_data.inputH;
    uint32_t inputW = tiling_data.inputW;
    uint32_t outputH = tiling_data.outputH;
    uint32_t outputW = tiling_data.outputW;

    uint32_t totalNC = batchSize * channels;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x);
    yGm.SetGlobalBuffer((__gm__ float*)y);

    uint32_t inputHW = inputH * inputW;
    uint32_t outputHW = outputH * outputW;

    // Each block processes a subset of (n, c) planes
    for (uint32_t nc = blockIdx; nc < totalNC; nc += blockNum) {
        uint32_t inOffset = nc * inputHW;
        uint32_t outOffset = nc * outputHW;

        for (uint32_t oh = 0; oh < outputH; oh++) {
            // src_y = (oh + 0.5) / 2.0 - 0.5 = oh * 0.5 - 0.25
            float srcY = (float)oh * 0.5f - 0.25f;
            if (srcY < 0.0f) srcY = 0.0f;
            if (srcY > (float)(inputH - 1)) srcY = (float)(inputH - 1);

            uint32_t y0 = (uint32_t)srcY;
            uint32_t y1 = y0 + 1;
            if (y1 >= inputH) y1 = inputH - 1;
            float yw = srcY - (float)y0;

            for (uint32_t ow = 0; ow < outputW; ow++) {
                float srcX = (float)ow * 0.5f - 0.25f;
                if (srcX < 0.0f) srcX = 0.0f;
                if (srcX > (float)(inputW - 1)) srcX = (float)(inputW - 1);

                uint32_t x0 = (uint32_t)srcX;
                uint32_t x1 = x0 + 1;
                if (x1 >= inputW) x1 = inputW - 1;
                float xw = srcX - (float)x0;

                float v00 = xGm.GetValue(inOffset + y0 * inputW + x0);
                float v01 = xGm.GetValue(inOffset + y0 * inputW + x1);
                float v10 = xGm.GetValue(inOffset + y1 * inputW + x0);
                float v11 = xGm.GetValue(inOffset + y1 * inputW + x1);

                float val = (1.0f - yw) * ((1.0f - xw) * v00 + xw * v01) +
                            yw * ((1.0f - xw) * v10 + xw * v11);

                yGm.SetValue(outOffset + oh * outputW + ow, val);
            }
        }
    }
}
