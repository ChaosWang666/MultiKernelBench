
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMean {
public:
    __aicore__ inline KernelMean() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t elementsPerBatch, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->elementsPerBatch = elementsPerBatch;
        this->tileNum = tileNum;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        // Each block processes a subset of batches
        this->batchStart = blockIdx * ((batchSize + blockNum - 1) / blockNum);
        this->batchEnd = (blockIdx + 1) * ((batchSize + blockNum - 1) / blockNum);
        if (this->batchEnd > batchSize) {
            this->batchEnd = batchSize;
        }
        this->batchCount = 0;
        if (this->batchEnd > this->batchStart) {
            this->batchCount = this->batchEnd - this->batchStart;
        }

        if (this->batchCount > 0) {
            // Determine tile length for processing each batch
            // We process elementsPerBatch for each batch in tiles
            uint32_t totalTiles = tileNum * BUFFER_NUM;
            this->tileLength = elementsPerBatch / totalTiles;
            // Align tileLength to 32 bytes (8 floats)
            if (this->tileLength < 8) {
                this->tileLength = 8;
            }
            this->tileLength = (this->tileLength / 8) * 8;
            this->totalTiles = elementsPerBatch / this->tileLength;
            this->tailLength = elementsPerBatch - this->totalTiles * this->tileLength;
            if (this->tailLength > 0 && this->tailLength < 8) {
                this->tailLength = 0;
                this->totalTiles = elementsPerBatch / this->tileLength;
            }

            xGm.SetGlobalBuffer((__gm__ float *)x + this->batchStart * elementsPerBatch,
                                 this->batchCount * elementsPerBatch);
            yGm.SetGlobalBuffer((__gm__ float *)y + this->batchStart, this->batchCount);

            uint32_t bufSize = this->tileLength * sizeof(float);
            if (this->tailLength > 0) {
                uint32_t tailBufSize = ((this->tailLength + 7) / 8) * 8 * sizeof(float);
                if (tailBufSize > bufSize) bufSize = tailBufSize;
            }
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize);
            pipe.InitBuffer(outQueueZ, 1, 8 * sizeof(float)); // small buffer for partial sum
        }
    }

    __aicore__ inline void Process()
    {
        if (this->batchCount == 0) return;

        for (uint32_t b = 0; b < this->batchCount; b++) {
            float batchSum = 0.0f;
            uint32_t batchOffset = b * this->elementsPerBatch;

            for (uint32_t t = 0; t < this->totalTiles; t++) {
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[batchOffset + t * this->tileLength], this->tileLength);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xProc = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> sumLocal = outQueueZ.AllocTensor<float>();
                AscendC::ReduceSum(sumLocal, xProc, inQueueX.GetTensorBuf(), this->tileLength);
                float tileSum = sumLocal.GetValue(0);
                batchSum += tileSum;
                outQueueZ.FreeTensor(sumLocal);
                inQueueX.FreeTensor(xProc);
            }

            // Handle tail
            if (this->tailLength > 0) {
                uint32_t alignedTail = ((this->tailLength + 7) / 8) * 8;
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[batchOffset + this->totalTiles * this->tileLength], alignedTail);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xProc = inQueueX.DeQue<float>();
                // Zero out padding
                for (uint32_t i = this->tailLength; i < alignedTail; i++) {
                    xProc.SetValue(i, 0.0f);
                }
                AscendC::LocalTensor<float> sumLocal = outQueueZ.AllocTensor<float>();
                AscendC::ReduceSum(sumLocal, xProc, inQueueX.GetTensorBuf(), alignedTail);
                float tileSum = sumLocal.GetValue(0);
                batchSum += tileSum;
                outQueueZ.FreeTensor(sumLocal);
                inQueueX.FreeTensor(xProc);
            }

            float meanVal = batchSum / (float)this->elementsPerBatch;
            // Write scalar to global memory
            AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
            outLocal.SetValue(0, meanVal);
            outQueueZ.EnQue(outLocal);
            AscendC::LocalTensor<float> outResult = outQueueZ.DeQue<float>();
            AscendC::DataCopy(yGm[b], outResult, 8);
            outQueueZ.FreeTensor(outResult);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t elementsPerBatch;
    uint32_t tileNum;
    uint32_t batchStart;
    uint32_t batchEnd;
    uint32_t batchCount;
    uint32_t tileLength;
    uint32_t totalTiles;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_mean_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMean op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.elementsPerBatch, tiling_data.tileNum);
    op.Process();
}
