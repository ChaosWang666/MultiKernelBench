
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void cumsum_exclusive_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t dimSize = tiling_data.dimSize;
    uint32_t outerSize = tiling_data.outerSize;
    uint32_t innerSize = tiling_data.innerSize;
    uint32_t numLines = outerSize * innerSize;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, outerSize * dimSize * innerSize);
    yGm.SetGlobalBuffer((__gm__ float*)y, outerSize * dimSize * innerSize);

    // Each block processes a subset of the lines (outer * inner combinations)
    for (uint32_t lineIdx = blockIdx; lineIdx < numLines; lineIdx += blockNum) {
        uint32_t outerIdx = lineIdx / innerSize;
        uint32_t innerIdx = lineIdx % innerSize;

        // Base offset for this line in global memory
        // Layout: [outerSize, dimSize, innerSize]
        // Element [o, d, i] is at offset o * dimSize * innerSize + d * innerSize + i
        uint32_t baseOffset = outerIdx * dimSize * innerSize + innerIdx;

        // Exclusive cumsum: y[0] = 0, y[i] = x[0] + x[1] + ... + x[i-1]
        if (innerSize == 1) {
            // Contiguous case along the cumsum dimension - can potentially vectorize
            // But for correctness, do scalar sequential scan
            float acc = 0.0f;
            for (uint32_t d = 0; d < dimSize; d++) {
                uint32_t offset = baseOffset + d * innerSize;
                float val = xGm.GetValue(offset);
                yGm.SetValue(offset, acc);
                acc += val;
            }
        } else {
            // Strided case
            float acc = 0.0f;
            for (uint32_t d = 0; d < dimSize; d++) {
                uint32_t offset = baseOffset + d * innerSize;
                float val = xGm.GetValue(offset);
                yGm.SetValue(offset, acc);
                acc += val;
            }
        }
    }
}
