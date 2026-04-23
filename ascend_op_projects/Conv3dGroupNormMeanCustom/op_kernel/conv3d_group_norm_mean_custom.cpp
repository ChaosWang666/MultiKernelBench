
#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 8192;
constexpr int32_t BUFFER_NUM = 2;

class KernelMean {
public:
    __aicore__ inline KernelMean() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t elementsPerBatch, float invElementsPerBatch)
    {
        this->batchSize = batchSize;
        this->elementsPerBatch = elementsPerBatch;
        this->invElementsPerBatch = invElementsPerBatch;

        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        if (blockNum == 0) blockNum = 1;
        uint32_t batchesPerCore = (batchSize + blockNum - 1) / blockNum;
        uint32_t sb = blockId * batchesPerCore;
        uint32_t eb = sb + batchesPerCore;
        if (sb > batchSize) sb = batchSize;
        if (eb > batchSize) eb = batchSize;
        this->startBatch = sb;
        this->endBatch = eb;

        xGm.SetGlobalBuffer((__gm__ float *)x, (uint64_t)batchSize * elementsPerBatch);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 32);
        pipe.InitBuffer(reduceBuf, 8 * 1024);
        pipe.InitBuffer(scalarBuf, 32);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t b = this->startBatch; b < this->endBatch; b++) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx)
    {
        uint64_t batchOffset = (uint64_t)batchIdx * elementsPerBatch;
        uint32_t numTiles = (elementsPerBatch + TILE_SIZE - 1) / TILE_SIZE;

        float totalSum = 0.0f;
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t tileOffset = t * TILE_SIZE;
            uint32_t tileLen = TILE_SIZE;
            if (tileOffset + TILE_SIZE > elementsPerBatch) {
                tileLen = elementsPerBatch - tileOffset;
            }

            // CopyIn
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams{1, (uint32_t)(tileLen * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(xLocal, xGm[batchOffset + tileOffset], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            // Compute
            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::ReduceSum<float, true>(scalarLocal, xIn, reduceTmp, (int32_t)tileLen);
            float tileSum = scalarLocal.GetValue(0);
            totalSum += tileSum;
            inQueueX.FreeTensor(xIn);
        }

        float mean = totalSum * invElementsPerBatch;

        AscendC::LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
        AscendC::Duplicate<float>(outLocal, mean, 8);
        outQueueY.EnQue(outLocal);
        AscendC::LocalTensor<float> y = outQueueY.DeQue<float>();

        AscendC::DataCopyExtParams outParams{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(yGm[batchIdx], y, outParams);
        outQueueY.FreeTensor(y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t elementsPerBatch;
    float invElementsPerBatch;
    uint32_t startBatch;
    uint32_t endBatch;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_mean_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMean op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.elementsPerBatch, tiling_data.invElementsPerBatch);
    op.Process();
}
