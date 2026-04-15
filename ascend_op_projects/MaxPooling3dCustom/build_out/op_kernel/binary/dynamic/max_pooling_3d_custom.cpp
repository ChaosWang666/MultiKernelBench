
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void max_pooling_3d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t dim1 = tiling_data.dim1;
    uint32_t dim2 = tiling_data.dim2;
    uint32_t dim3 = tiling_data.dim3;
    uint32_t outDim1 = tiling_data.outDim1;
    uint32_t outDim2 = tiling_data.outDim2;
    uint32_t outDim3 = tiling_data.outDim3;
    uint32_t kernelSize = tiling_data.kernelSize;
    uint32_t stride = tiling_data.stride;
    uint32_t padding = tiling_data.padding;
    uint32_t totalOutputElements = tiling_data.totalOutputElements;
    uint32_t blockDim = tiling_data.blockDim;

    uint32_t blockIdx = AscendC::GetBlockIdx();

    // Each block processes a portion of output elements
    uint32_t elementsPerBlock = (totalOutputElements + blockDim - 1) / blockDim;
    uint32_t startIdx = blockIdx * elementsPerBlock;
    uint32_t endIdx = startIdx + elementsPerBlock;
    if (endIdx > totalOutputElements) {
        endIdx = totalOutputElements;
    }

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * channels * dim1 * dim2 * dim3);
    yGm.SetGlobalBuffer((__gm__ float *)y, totalOutputElements);

    uint32_t outSpatial = outDim1 * outDim2 * outDim3;
    uint32_t inSpatial = dim1 * dim2 * dim3;

    for (uint32_t idx = startIdx; idx < endIdx; idx++) {
        // Decode output index: idx -> (b, c, od1, od2, od3)
        uint32_t remaining = idx;
        uint32_t b = remaining / (channels * outSpatial);
        remaining = remaining % (channels * outSpatial);
        uint32_t c = remaining / outSpatial;
        remaining = remaining % outSpatial;
        uint32_t od1 = remaining / (outDim2 * outDim3);
        remaining = remaining % (outDim2 * outDim3);
        uint32_t od2 = remaining / outDim3;
        uint32_t od3 = remaining % outDim3;

        // Compute input window start
        int32_t id1_start = (int32_t)(od1 * stride) - (int32_t)padding;
        int32_t id2_start = (int32_t)(od2 * stride) - (int32_t)padding;
        int32_t id3_start = (int32_t)(od3 * stride) - (int32_t)padding;

        float maxVal = -3.402823e+38f; // -FLT_MAX

        uint32_t baseOffset = (b * channels + c) * inSpatial;

        for (uint32_t kd1 = 0; kd1 < kernelSize; kd1++) {
            int32_t id1 = id1_start + (int32_t)kd1;
            if (id1 < 0 || id1 >= (int32_t)dim1) continue;
            for (uint32_t kd2 = 0; kd2 < kernelSize; kd2++) {
                int32_t id2 = id2_start + (int32_t)kd2;
                if (id2 < 0 || id2 >= (int32_t)dim2) continue;
                for (uint32_t kd3 = 0; kd3 < kernelSize; kd3++) {
                    int32_t id3 = id3_start + (int32_t)kd3;
                    if (id3 < 0 || id3 >= (int32_t)dim3) continue;
                    uint32_t inIdx = baseOffset + (uint32_t)id1 * dim2 * dim3 + (uint32_t)id2 * dim3 + (uint32_t)id3;
                    float val = xGm.GetValue(inIdx);
                    if (val > maxVal) {
                        maxVal = val;
                    }
                }
            }
        }

        yGm.SetValue(idx, maxVal);
    }
}
