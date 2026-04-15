
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void trilinear_upsample_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t iD = tiling_data.inputDepth;
    uint32_t iH = tiling_data.inputHeight;
    uint32_t iW = tiling_data.inputWidth;
    uint32_t oD = tiling_data.outputDepth;
    uint32_t oH = tiling_data.outputHeight;
    uint32_t oW = tiling_data.outputWidth;

    uint32_t totalBC = batchSize * channels;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    uint32_t inputSliceSize = iD * iH * iW;
    uint32_t outputSliceSize = oD * oH * oW;

    __gm__ float* xGm = (__gm__ float*)x;
    __gm__ float* yGm = (__gm__ float*)y;

    // Each block processes some (batch, channel) slices
    for (uint32_t bc = blockIdx; bc < totalBC; bc += blockNum) {
        __gm__ float* inSlice = xGm + bc * inputSliceSize;
        __gm__ float* outSlice = yGm + bc * outputSliceSize;

        for (uint32_t od = 0; od < oD; od++) {
            // align_corners: map output coord to input coord
            // real_id = od * (iD - 1) / (oD - 1)
            float rd;
            if (oD > 1) {
                rd = (float)od * (float)(iD - 1) / (float)(oD - 1);
            } else {
                rd = 0.0f;
            }
            uint32_t id0 = (uint32_t)rd;
            uint32_t id1 = id0 + 1;
            if (id1 >= iD) id1 = iD - 1;
            float wd1 = rd - (float)id0;
            float wd0 = 1.0f - wd1;

            for (uint32_t oh = 0; oh < oH; oh++) {
                float rh;
                if (oH > 1) {
                    rh = (float)oh * (float)(iH - 1) / (float)(oH - 1);
                } else {
                    rh = 0.0f;
                }
                uint32_t ih0 = (uint32_t)rh;
                uint32_t ih1 = ih0 + 1;
                if (ih1 >= iH) ih1 = iH - 1;
                float wh1 = rh - (float)ih0;
                float wh0 = 1.0f - wh1;

                for (uint32_t ow = 0; ow < oW; ow++) {
                    float rw;
                    if (oW > 1) {
                        rw = (float)ow * (float)(iW - 1) / (float)(oW - 1);
                    } else {
                        rw = 0.0f;
                    }
                    uint32_t iw0 = (uint32_t)rw;
                    uint32_t iw1 = iw0 + 1;
                    if (iw1 >= iW) iw1 = iW - 1;
                    float ww1 = rw - (float)iw0;
                    float ww0 = 1.0f - ww1;

                    // Trilinear interpolation: 8 corners
                    float v000 = *(inSlice + id0 * iH * iW + ih0 * iW + iw0);
                    float v001 = *(inSlice + id0 * iH * iW + ih0 * iW + iw1);
                    float v010 = *(inSlice + id0 * iH * iW + ih1 * iW + iw0);
                    float v011 = *(inSlice + id0 * iH * iW + ih1 * iW + iw1);
                    float v100 = *(inSlice + id1 * iH * iW + ih0 * iW + iw0);
                    float v101 = *(inSlice + id1 * iH * iW + ih0 * iW + iw1);
                    float v110 = *(inSlice + id1 * iH * iW + ih1 * iW + iw0);
                    float v111 = *(inSlice + id1 * iH * iW + ih1 * iW + iw1);

                    float val = wd0 * wh0 * ww0 * v000 +
                                wd0 * wh0 * ww1 * v001 +
                                wd0 * wh1 * ww0 * v010 +
                                wd0 * wh1 * ww1 * v011 +
                                wd1 * wh0 * ww0 * v100 +
                                wd1 * wh0 * ww1 * v101 +
                                wd1 * wh1 * ww0 * v110 +
                                wd1 * wh1 * ww1 * v111;

                    uint32_t outIdx = od * oH * oW + oh * oW + ow;
                    *(outSlice + outIdx) = val;
                }
            }
        }
    }
}
