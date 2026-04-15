
#include "kernel_operator.h"

// Bicubic interpolation kernel for NCHW tensor upsampling with align_corners=True
// Each AI core processes a subset of the total output pixels across all (batch, channel) planes.

__aicore__ inline float cubicWeight(float x) {
    // Keys cubic with a = -0.75
    const float A = -0.75f;
    float absX = x < 0.0f ? -x : x;
    if (absX <= 1.0f) {
        return ((A + 2.0f) * absX - (A + 3.0f)) * absX * absX + 1.0f;
    } else if (absX < 2.0f) {
        return ((A * absX - 5.0f * A) * absX + 8.0f * A) * absX - 4.0f * A;
    }
    return 0.0f;
}

extern "C" __global__ __aicore__ void bicubic_upsample_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t inH = tiling_data.inputHeight;
    uint32_t inW = tiling_data.inputWidth;
    uint32_t outH = tiling_data.outputHeight;
    uint32_t outW = tiling_data.outputWidth;

    uint32_t totalPlanes = batchSize * channels;  // number of (batch, channel) planes
    uint32_t outPixelsPerPlane = outH * outW;
    uint32_t inPixelsPerPlane = inH * inW;

    // Total work items: totalPlanes * outPixelsPerPlane
    uint64_t totalWork = (uint64_t)totalPlanes * (uint64_t)outPixelsPerPlane;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    // Compute work range for this block
    uint64_t workPerBlock = (totalWork + blockNum - 1) / blockNum;
    uint64_t startWork = (uint64_t)blockIdx * workPerBlock;
    uint64_t endWork = startWork + workPerBlock;
    if (endWork > totalWork) endWork = totalWork;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inH * inW);
    yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channels * outH * outW);

    // Precompute scale factors for align_corners=True
    float scaleH = (outH > 1) ? (float)(inH - 1) / (float)(outH - 1) : 0.0f;
    float scaleW = (outW > 1) ? (float)(inW - 1) / (float)(outW - 1) : 0.0f;

    for (uint64_t idx = startWork; idx < endWork; idx++) {
        uint32_t planeIdx = (uint32_t)(idx / outPixelsPerPlane);
        uint32_t pixelIdx = (uint32_t)(idx % outPixelsPerPlane);
        uint32_t oh = pixelIdx / outW;
        uint32_t ow = pixelIdx % outW;

        // Map output pixel to input coordinate
        float inY = scaleH * (float)oh;
        float inX = scaleW * (float)ow;

        int32_t iy = (int32_t)inY;
        int32_t ix = (int32_t)inX;
        float fy = inY - (float)iy;
        float fx = inX - (float)ix;

        uint32_t inPlaneOffset = planeIdx * inPixelsPerPlane;
        uint32_t outPlaneOffset = planeIdx * outPixelsPerPlane;

        float result = 0.0f;

        for (int32_t j = -1; j <= 2; j++) {
            float wy = cubicWeight(fy - (float)j);
            int32_t srcY = iy + j;
            // Clamp
            if (srcY < 0) srcY = 0;
            if (srcY >= (int32_t)inH) srcY = (int32_t)inH - 1;

            for (int32_t i = -1; i <= 2; i++) {
                float wx = cubicWeight(fx - (float)i);
                int32_t srcX = ix + i;
                // Clamp
                if (srcX < 0) srcX = 0;
                if (srcX >= (int32_t)inW) srcX = (int32_t)inW - 1;

                float val = xGm.GetValue(inPlaneOffset + (uint32_t)srcY * inW + (uint32_t)srcX);
                result += wy * wx * val;
            }
        }

        yGm.SetValue(outPlaneOffset + oh * outW + ow, result);
    }
}
