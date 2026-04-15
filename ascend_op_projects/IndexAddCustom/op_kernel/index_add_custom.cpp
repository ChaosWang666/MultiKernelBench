
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelIndexAdd {
public:
    __aicore__ inline KernelIndexAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR values, GM_ADDR z,
                                 uint32_t xRows, uint32_t cols, uint32_t numIndices)
    {
        this->xRows = xRows;
        this->cols = cols;
        this->numIndices = numIndices;

        uint32_t totalX = xRows * cols;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalX);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, numIndices);
        valuesGm.SetGlobalBuffer((__gm__ float *)values, numIndices * cols);
        zGm.SetGlobalBuffer((__gm__ float *)z, totalX);

        // Determine tile size for cols dimension
        // We process one row at a time, tiling along cols
        // Use up to 32KB per buffer => 8192 floats max
        uint32_t maxTileLen = 4096; // conservative
        if (cols <= maxTileLen) {
            this->tileLength = cols;
            this->tileNum = 1;
        } else {
            this->tileLength = maxTileLen;
            this->tileNum = (cols + maxTileLen - 1) / maxTileLen;
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // First copy x to z
        for (uint32_t row = 0; row < xRows; row++) {
            for (uint32_t t = 0; t < tileNum; t++) {
                uint32_t offset = row * cols + t * tileLength;
                uint32_t curLen = tileLength;
                if (t == tileNum - 1 && (cols % tileLength) != 0) {
                    curLen = cols - t * tileLength;
                }
                // Align curLen to 8 for DataCopy (32 bytes = 8 floats)
                uint32_t alignedLen = (curLen + 7) / 8 * 8;
                if (alignedLen > tileLength) alignedLen = tileLength;

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[offset], alignedLen);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xLocal2 = inQueueX.DeQue<float>();
                AscendC::DataCopy(zGm[offset], xLocal2, alignedLen);
                inQueueX.FreeTensor(xLocal2);
            }
        }

        // Now for each index, add values row to z row
        for (uint32_t i = 0; i < numIndices; i++) {
            // Read index - we need to copy at least 8 int32s (32 bytes)
            // We'll read a small batch and pick the first element
            int32_t idx;
            // Use a simple approach: copy indices one aligned block at a time
            // For simplicity, process index by reading from GM
            // We need to get the index value. Copy 8 int32_t values.
            {
                AscendC::LocalTensor<float> tmpLocal = inQueueX.AllocTensor<float>();
                // Reinterpret as int32 by copying from indicesGm
                // Align i to multiple of 8
                uint32_t alignedIdx = (i / 8) * 8;
                AscendC::DataCopy(tmpLocal, ((__gm__ float*)((__gm__ int32_t*)indicesGm.GetPhyAddr()))[alignedIdx], 8);
                inQueueX.EnQue(tmpLocal);
                AscendC::LocalTensor<float> tmpLocal2 = inQueueX.DeQue<float>();
                // Get the int32 at position (i - alignedIdx)
                int32_t* tmpPtr = (int32_t*)tmpLocal2.GetPhyAddr();
                idx = tmpPtr[i - alignedIdx];
                inQueueX.FreeTensor(tmpLocal2);
            }

            for (uint32_t t = 0; t < tileNum; t++) {
                uint32_t colOffset = t * tileLength;
                uint32_t curLen = tileLength;
                if (t == tileNum - 1 && (cols % tileLength) != 0) {
                    curLen = cols - t * tileLength;
                }
                uint32_t alignedLen = (curLen + 7) / 8 * 8;
                if (alignedLen > tileLength) alignedLen = tileLength;

                uint32_t zOffset = (uint32_t)idx * cols + colOffset;
                uint32_t vOffset = i * cols + colOffset;

                // Load z row tile
                AscendC::LocalTensor<float> zLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(zLocal, zGm[zOffset], alignedLen);
                inQueueX.EnQue(zLocal);

                // Load values row tile
                AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
                AscendC::DataCopy(vLocal, valuesGm[vOffset], alignedLen);
                inQueueV.EnQue(vLocal);

                // Add
                AscendC::LocalTensor<float> zLocal2 = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> vLocal2 = inQueueV.DeQue<float>();
                AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
                AscendC::Add(outLocal, zLocal2, vLocal2, alignedLen);
                outQueueZ.EnQue(outLocal);
                inQueueX.FreeTensor(zLocal2);
                inQueueV.FreeTensor(vLocal2);

                // Write back
                AscendC::LocalTensor<float> outLocal2 = outQueueZ.DeQue<float>();
                AscendC::DataCopy(zGm[zOffset], outLocal2, alignedLen);
                outQueueZ.FreeTensor(outLocal2);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<float> valuesGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t xRows;
    uint32_t cols;
    uint32_t numIndices;
    uint32_t tileLength;
    uint32_t tileNum;
};

extern "C" __global__ __aicore__ void index_add_custom(GM_ADDR x, GM_ADDR indices, GM_ADDR values, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelIndexAdd op;
    op.Init(x, indices, values, z, tiling_data.xRows, tiling_data.cols, tiling_data.numIndices);
    op.Process();
}
