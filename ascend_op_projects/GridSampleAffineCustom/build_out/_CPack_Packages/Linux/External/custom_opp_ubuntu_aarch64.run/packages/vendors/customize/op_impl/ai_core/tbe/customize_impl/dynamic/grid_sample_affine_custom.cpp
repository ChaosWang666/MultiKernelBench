
#include "kernel_operator.h"

// Grid sample with affine transform, bilinear interpolation, align_corners=False
// Input x: (N, C, H, W), theta: (N, 2, 3)
// Output y: (N, C, H, W)
// Each block processes one batch element (blockIdx maps to batch index)

extern "C" __global__ __aicore__ void grid_sample_affine_custom(GM_ADDR x, GM_ADDR theta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t inH = tiling_data.inH;
    uint32_t inW = tiling_data.inW;
    uint32_t outH = tiling_data.outH;
    uint32_t outW = tiling_data.outW;
    uint32_t tileNum = tiling_data.tileNum;
    
    uint32_t batchIdx = AscendC::GetBlockIdx();
    if (batchIdx >= batchSize) return;
    
    // Pointers for this batch
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> thetaGm;
    AscendC::GlobalTensor<float> yGm;
    
    uint32_t spatialSize = inH * inW;
    uint32_t batchElements = channels * spatialSize;
    
    xGm.SetGlobalBuffer((__gm__ float*)x + batchIdx * batchElements, batchElements);
    thetaGm.SetGlobalBuffer((__gm__ float*)theta + batchIdx * 6, 6);
    yGm.SetGlobalBuffer((__gm__ float*)y + batchIdx * batchElements, batchElements);
    
    // Read theta values using pipe
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> thetaBuf;
    // Align to 32 bytes = 8 floats
    pipe.InitBuffer(thetaBuf, 8 * sizeof(float));
    AscendC::LocalTensor<float> thetaLocal = thetaBuf.Get<float>();
    // Copy 8 floats (only 6 used, but must be aligned to 32 bytes)
    AscendC::DataCopy(thetaLocal, thetaGm, 8);
    AscendC::PipeBarrier<PIPE_ALL>();

    float a00 = thetaLocal.GetValue(0);
    float a01 = thetaLocal.GetValue(1);
    float a02 = thetaLocal.GetValue(2);
    float a10 = thetaLocal.GetValue(3);
    float a11 = thetaLocal.GetValue(4);
    float a12 = thetaLocal.GetValue(5);
    
    // For each output pixel (oh, ow), compute grid coordinates
    // Normalized output coords: gx_n = 2*ow/(outW) - 1 + 1/outW (align_corners=False)
    //                           gy_n = 2*oh/(outH) - 1 + 1/outH
    // Then apply affine: src_x_n = a00*gx_n + a01*gy_n + a02
    //                     src_y_n = a10*gx_n + a11*gy_n + a12
    // Unnormalize: src_x = (src_x_n + 1) * inW / 2 - 0.5 (align_corners=False)
    //              src_y = (src_y_n + 1) * inH / 2 - 0.5

    // Process output pixels in tiles across channels
    // We process one output pixel at a time, all channels
    // Use a small tile buffer for channel data

    uint32_t tileSizeC = 64; // process 64 channels at a time (256 bytes aligned)
    if (tileSizeC > channels) {
        // Round up to nearest multiple of 8 for alignment
        tileSizeC = (channels + 7) / 8 * 8;
    }

    AscendC::TBuf<AscendC::TPosition::VECCALC> buf0, buf1, buf2, buf3, buf4;
    pipe.InitBuffer(buf0, tileSizeC * sizeof(float));
    pipe.InitBuffer(buf1, tileSizeC * sizeof(float));
    pipe.InitBuffer(buf2, tileSizeC * sizeof(float));
    pipe.InitBuffer(buf3, tileSizeC * sizeof(float));
    pipe.InitBuffer(buf4, tileSizeC * sizeof(float));

    AscendC::LocalTensor<float> topLeft = buf0.Get<float>();
    AscendC::LocalTensor<float> topRight = buf1.Get<float>();
    AscendC::LocalTensor<float> botLeft = buf2.Get<float>();
    AscendC::LocalTensor<float> botRight = buf3.Get<float>();
    AscendC::LocalTensor<float> result = buf4.Get<float>();

    float outW_f = (float)outW;
    float outH_f = (float)outH;
    float inW_f = (float)inW;
    float inH_f = (float)inH;

    for (uint32_t oh = 0; oh < outH; oh++) {
        float gy_n = 2.0f * ((float)oh + 0.5f) / outH_f - 1.0f;
        for (uint32_t ow = 0; ow < outW; ow++) {
            float gx_n = 2.0f * ((float)ow + 0.5f) / outW_f - 1.0f;
            
            // Apply affine
            float src_x_n = a00 * gx_n + a01 * gy_n + a02;
            float src_y_n = a10 * gx_n + a11 * gy_n + a12;
            
            // Unnormalize (align_corners=False)
            float src_x = (src_x_n + 1.0f) * inW_f * 0.5f - 0.5f;
            float src_y = (src_y_n + 1.0f) * inH_f * 0.5f - 0.5f;
            
            int32_t x0 = (int32_t)((src_x >= 0) ? src_x : (src_x - 1.0f));
            int32_t y0 = (int32_t)((src_y >= 0) ? src_y : (src_y - 1.0f));
            int32_t x1 = x0 + 1;
            int32_t y1 = y0 + 1;
            
            float wx1 = src_x - (float)x0;
            float wx0 = 1.0f - wx1;
            float wy1 = src_y - (float)y0;
            float wy0 = 1.0f - wy1;
            
            float w00 = wy0 * wx0;
            float w01 = wy0 * wx1;
            float w10 = wy1 * wx0;
            float w11 = wy1 * wx1;
            
            bool v00 = (x0 >= 0 && x0 < (int32_t)inW && y0 >= 0 && y0 < (int32_t)inH);
            bool v01 = (x1 >= 0 && x1 < (int32_t)inW && y0 >= 0 && y0 < (int32_t)inH);
            bool v10 = (x0 >= 0 && x0 < (int32_t)inW && y1 >= 0 && y1 < (int32_t)inH);
            bool v11 = (x1 >= 0 && x1 < (int32_t)inW && y1 >= 0 && y1 < (int32_t)inH);
            
            uint32_t outPixelOffset = oh * outW + ow;
            
            // Process channels in tiles
            for (uint32_t cStart = 0; cStart < channels; cStart += tileSizeC) {
                uint32_t cEnd = cStart + tileSizeC;
                if (cEnd > channels) cEnd = channels;
                uint32_t cLen = cEnd - cStart;
                uint32_t cLenAligned = (cLen + 7) / 8 * 8;
                
                // Zero out result
                AscendC::Duplicate<float>(result, 0.0f, cLenAligned);
                AscendC::PipeBarrier<PIPE_V>();
                
                if (v00 && w00 != 0.0f) {
                    uint32_t srcOff00 = y0 * inW + x0;
                    // Load channel values for top-left
                    for (uint32_t ci = 0; ci < cLen; ci++) {
                        uint32_t c = cStart + ci;
                        float val = xGm.GetValue(c * spatialSize + srcOff00);
                        topLeft.SetValue(ci, val);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls<float>(topLeft, topLeft, w00, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(result, result, topLeft, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                
                if (v01 && w01 != 0.0f) {
                    uint32_t srcOff01 = y0 * inW + x1;
                    for (uint32_t ci = 0; ci < cLen; ci++) {
                        uint32_t c = cStart + ci;
                        float val = xGm.GetValue(c * spatialSize + srcOff01);
                        topRight.SetValue(ci, val);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls<float>(topRight, topRight, w01, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(result, result, topRight, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                
                if (v10 && w10 != 0.0f) {
                    uint32_t srcOff10 = y1 * inW + x0;
                    for (uint32_t ci = 0; ci < cLen; ci++) {
                        uint32_t c = cStart + ci;
                        float val = xGm.GetValue(c * spatialSize + srcOff10);
                        botLeft.SetValue(ci, val);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls<float>(botLeft, botLeft, w10, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(result, result, botLeft, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                
                if (v11 && w11 != 0.0f) {
                    uint32_t srcOff11 = y1 * inW + x1;
                    for (uint32_t ci = 0; ci < cLen; ci++) {
                        uint32_t c = cStart + ci;
                        float val = xGm.GetValue(c * spatialSize + srcOff11);
                        botRight.SetValue(ci, val);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls<float>(botRight, botRight, w11, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float>(result, result, botRight, cLenAligned);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                
                // Write result for these channels
                for (uint32_t ci = 0; ci < cLen; ci++) {
                    uint32_t c = cStart + ci;
                    float val = result.GetValue(ci);
                    yGm.SetValue(c * spatialSize + outPixelOffset, val);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
}
