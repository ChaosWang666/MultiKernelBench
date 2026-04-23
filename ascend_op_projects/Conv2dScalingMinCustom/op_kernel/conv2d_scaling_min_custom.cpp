
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dScalingMin {
public:
    __aicore__ inline KernelConv2dScalingMin() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize,
                                 uint32_t channels, uint32_t spatialSize,
                                 uint32_t tileLen, float scaleFactor)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->spatialSize = spatialSize;
        this->tileLen = tileLen;
        this->scaleFactor = scaleFactor;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t batchesPerBlock = (batchSize + blockNum - 1) / blockNum;
        uint32_t sBatch = blockIdx * batchesPerBlock;
        uint32_t eBatch = sBatch + batchesPerBlock;
        if (sBatch >= batchSize) {
            this->startBatch = batchSize;
            this->numBatches = 0;
        } else {
            if (eBatch > batchSize) eBatch = batchSize;
            this->startBatch = sBatch;
            this->numBatches = eBatch - sBatch;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * channels * spatialSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * spatialSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(accBuf, tileLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (numBatches == 0) return;

        for (uint32_t b = 0; b < numBatches; b++) {
            uint32_t batchIdx = startBatch + b;
            uint32_t numTiles = (spatialSize + tileLen - 1) / tileLen;
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t offset = t * tileLen;
                uint32_t remaining = spatialSize - offset;
                uint32_t curLen = (remaining < tileLen) ? remaining : tileLen;
                ProcessTile(batchIdx, offset, curLen);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t batchIdx, uint32_t spatialOffset, uint32_t curLen)
    {
        AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
        uint64_t batchBase = (uint64_t)batchIdx * (uint64_t)channels * (uint64_t)spatialSize;

        for (uint32_t c = 0; c < channels; c++) {
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curLen * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0.0f;

            uint64_t srcOffset = batchBase + (uint64_t)c * (uint64_t)spatialSize + (uint64_t)spatialOffset;
            AscendC::DataCopyPad(xLocal, xGm[srcOffset], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            if (c == 0) {
                AscendC::Adds<float>(accLocal, xIn, 0.0f, curLen);
            } else {
                AscendC::Min<float>(accLocal, accLocal, xIn, curLen);
            }
            inQueueX.FreeTensor(xIn);
        }

        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Muls<float>(yLocal, accLocal, scaleFactor, curLen);
        outQueueY.EnQue<float>(yLocal);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        uint64_t dstOffset = (uint64_t)batchIdx * (uint64_t)spatialSize + (uint64_t)spatialOffset;
        AscendC::DataCopyPad(yGm[dstOffset], yOut, copyOutParams);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t spatialSize;
    uint32_t tileLen;
    uint32_t startBatch;
    uint32_t numBatches;
    float scaleFactor;
};

extern "C" __global__ __aicore__ void conv2d_scaling_min_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dScalingMin op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.channels,
            tiling_data.spatialSize, tiling_data.tileLen, tiling_data.scaleFactor);
    op.Process();
}
