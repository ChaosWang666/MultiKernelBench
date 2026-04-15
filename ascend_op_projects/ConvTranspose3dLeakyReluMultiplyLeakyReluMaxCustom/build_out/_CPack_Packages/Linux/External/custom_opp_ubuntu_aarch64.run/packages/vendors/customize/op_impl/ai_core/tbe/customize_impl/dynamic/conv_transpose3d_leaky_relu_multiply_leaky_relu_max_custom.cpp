
#include "kernel_operator.h"

constexpr float NEGATIVE_SLOPE = 0.2f;

// Fused kernel: LeakyReLU -> Multiply -> LeakyReLU -> MaxPool3d(2)
// Input x: [N, C, D, H, W], multiplier: [C], Output: [N, C, D/2, H/2, W/2]
// Each block processes a subset of (n, c) pairs. For each (n, c), we do the full D, H, W processing.

extern "C" __global__ __aicore__ void conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom(
    GM_ADDR x, GM_ADDR multiplier, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);

    uint32_t N = t.batchSize;
    uint32_t C = t.channels;
    uint32_t D = t.depthIn;
    uint32_t H = t.heightIn;
    uint32_t W = t.widthIn;
    uint32_t Do = t.depthOut;
    uint32_t Ho = t.heightOut;
    uint32_t Wo = t.widthOut;

    uint32_t totalNC = N * C;
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    uint32_t spatialIn = D * H * W;
    uint32_t spatialOut = Do * Ho * Wo;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> mulGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, N * C * spatialIn);
    mulGm.SetGlobalBuffer((__gm__ float*)multiplier, C);
    yGm.SetGlobalBuffer((__gm__ float*)y, N * C * spatialOut);

    // Each block iterates over nc indices assigned to it
    for (uint32_t nc = blockIdx; nc < totalNC; nc += blockNum) {
        uint32_t c = nc % C;
        float mulVal = mulGm.GetValue(c);

        uint32_t inOffset = nc * spatialIn;
        uint32_t outOffset = nc * spatialOut;

        // Process each output element
        for (uint32_t od = 0; od < Do; od++) {
            for (uint32_t oh = 0; oh < Ho; oh++) {
                for (uint32_t ow = 0; ow < Wo; ow++) {
                    float maxVal = -3.402823e+38f;
                    // 2x2x2 pooling window
                    for (uint32_t dd = 0; dd < 2; dd++) {
                        for (uint32_t dh = 0; dh < 2; dh++) {
                            for (uint32_t dw = 0; dw < 2; dw++) {
                                uint32_t id = od * 2 + dd;
                                uint32_t ih = oh * 2 + dh;
                                uint32_t iw = ow * 2 + dw;
                                uint32_t idx = inOffset + id * H * W + ih * W + iw;
                                float val = xGm.GetValue(idx);
                                // LeakyReLU
                                val = val >= 0.0f ? val : val * NEGATIVE_SLOPE;
                                // Multiply
                                val = val * mulVal;
                                // LeakyReLU 
                                val = val >= 0.0f ? val : val * NEGATIVE_SLOPE;
                                if (val > maxVal) {
                                    maxVal = val;
                                }
                            }
                        }
                    }
                    uint32_t outIdx = outOffset + od * Ho * Wo + oh * Wo + ow;
                    yGm.SetValue(outIdx, maxVal);
                }
            }
        }
    }
}
