
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void average_pooling_3d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t depth = tiling_data.depth;
    uint32_t height = tiling_data.height;
    uint32_t width = tiling_data.width;
    uint32_t outDepth = tiling_data.outDepth;
    uint32_t outHeight = tiling_data.outHeight;
    uint32_t outWidth = tiling_data.outWidth;
    uint32_t kernelSize = tiling_data.kernelSize;
    uint32_t stride = tiling_data.stride;
    uint32_t padding = tiling_data.padding;
    uint32_t totalOutputElements = tiling_data.totalOutputElements;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    uint32_t inSpatial = depth * height * width;
    uint32_t outSpatial = outDepth * outHeight * outWidth;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inSpatial);
    yGm.SetGlobalBuffer((__gm__ float*)y, totalOutputElements);

    float invKernelVol = 1.0f / (float)(kernelSize * kernelSize * kernelSize);

    // Each block processes a portion of the total output elements
    uint32_t elementsPerBlock = (totalOutputElements + blockNum - 1) / blockNum;
    uint32_t startElem = blockIdx * elementsPerBlock;
    uint32_t endElem = startElem + elementsPerBlock;
    if (endElem > totalOutputElements) endElem = totalOutputElements;

    for (uint32_t idx = startElem; idx < endElem; idx++) {
        // Decompose linear index into (n, c, od, oh, ow)
        uint32_t rem = idx;
        uint32_t ow = rem % outWidth;
        rem = rem / outWidth;
        uint32_t oh = rem % outHeight;
        rem = rem / outHeight;
        uint32_t od = rem % outDepth;
        rem = rem / outDepth;
        uint32_t c = rem % channels;
        uint32_t n = rem / channels;

        // Compute pooling window boundaries in input space
        int32_t d_start = (int32_t)(od * stride) - (int32_t)padding;
        int32_t h_start = (int32_t)(oh * stride) - (int32_t)padding;
        int32_t w_start = (int32_t)(ow * stride) - (int32_t)padding;
        int32_t d_end = d_start + (int32_t)kernelSize;
        int32_t h_end = h_start + (int32_t)kernelSize;
        int32_t w_end = w_start + (int32_t)kernelSize;

        // Clamp to input bounds
        int32_t d_start_c = d_start < 0 ? 0 : d_start;
        int32_t h_start_c = h_start < 0 ? 0 : h_start;
        int32_t w_start_c = w_start < 0 ? 0 : w_start;
        int32_t d_end_c = d_end > (int32_t)depth ? (int32_t)depth : d_end;
        int32_t h_end_c = h_end > (int32_t)height ? (int32_t)height : h_end;
        int32_t w_end_c = w_end > (int32_t)width ? (int32_t)width : w_end;

        float sum = 0.0f;
        uint32_t baseOffset = (n * channels + c) * inSpatial;

        for (int32_t dd = d_start_c; dd < d_end_c; dd++) {
            for (int32_t hh = h_start_c; hh < h_end_c; hh++) {
                for (int32_t ww = w_start_c; ww < w_end_c; ww++) {
                    uint32_t inIdx = baseOffset + (uint32_t)dd * height * width + (uint32_t)hh * width + (uint32_t)ww;
                    sum += xGm.GetValue(inIdx);
                }
            }
        }

        // count_include_pad=True by default: divide by full kernel volume
        float avg = sum * invKernelVol;
        yGm.SetValue(idx, avg);
    }
}
