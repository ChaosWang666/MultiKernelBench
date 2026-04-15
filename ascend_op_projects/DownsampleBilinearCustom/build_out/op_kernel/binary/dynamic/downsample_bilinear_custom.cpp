
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void downsample_bilinear_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t inputH = tiling_data.inputH;
    uint32_t inputW = tiling_data.inputW;
    uint32_t outputH = tiling_data.outputH;
    uint32_t outputW = tiling_data.outputW;

    uint32_t totalPlanes = batchSize * channels;
    uint32_t numBlocks = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();

    uint32_t inputPlaneSize = inputH * inputW;
    uint32_t outputPlaneSize = outputH * outputW;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inputPlaneSize);
    yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channels * outputPlaneSize);

    // Each block processes a subset of (batch, channel) planes
    for (uint32_t planeIdx = blockIdx; planeIdx < totalPlanes; planeIdx += numBlocks) {
        uint32_t inputOffset = planeIdx * inputPlaneSize;
        uint32_t outputOffset = planeIdx * outputPlaneSize;

        // Bilinear interpolation with align_corners=False
        // scale = input_size / output_size
        float scaleH = (float)inputH / (float)outputH;
        float scaleW = (float)inputW / (float)outputW;

        // Process each output row
        for (uint32_t oh = 0; oh < outputH; oh++) {
            // Compute source y coordinate
            float srcH = ((float)oh + 0.5f) * scaleH - 0.5f;
            int32_t h0 = (int32_t)srcH;
            if (srcH < 0.0f) h0 = h0 - 1;
            float hLerp = srcH - (float)h0;
            if (h0 < 0) { h0 = 0; hLerp = 0.0f; }
            int32_t h1 = h0 + 1;
            if (h1 >= (int32_t)inputH) h1 = (int32_t)inputH - 1;
            if (h0 >= (int32_t)inputH) h0 = (int32_t)inputH - 1;

            for (uint32_t ow = 0; ow < outputW; ow++) {
                float srcW = ((float)ow + 0.5f) * scaleW - 0.5f;
                int32_t w0 = (int32_t)srcW;
                if (srcW < 0.0f) w0 = w0 - 1;
                float wLerp = srcW - (float)w0;
                if (w0 < 0) { w0 = 0; wLerp = 0.0f; }
                int32_t w1 = w0 + 1;
                if (w1 >= (int32_t)inputW) w1 = (int32_t)inputW - 1;
                if (w0 >= (int32_t)inputW) w0 = (int32_t)inputW - 1;

                // Read 4 corners from global memory
                float v00 = xGm.GetValue(inputOffset + h0 * inputW + w0);
                float v01 = xGm.GetValue(inputOffset + h0 * inputW + w1);
                float v10 = xGm.GetValue(inputOffset + h1 * inputW + w0);
                float v11 = xGm.GetValue(inputOffset + h1 * inputW + w1);

                // Bilinear interpolation
                float val = v00 * (1.0f - hLerp) * (1.0f - wLerp) +
                            v01 * (1.0f - hLerp) * wLerp +
                            v10 * hLerp * (1.0f - wLerp) +
                            v11 * hLerp * wLerp;

                yGm.SetValue(outputOffset + oh * outputW + ow, val);
            }
        }
    }
}
