
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dMinAddMultiply {
public:
    __aicore__ inline KernelConv2dMinAddMultiply() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t totalPairs, uint32_t spatialSize,
                                 uint32_t channels, uint32_t tileSize,
                                 float constantValue, float scalingFactor)
    {
        this->totalPairs = totalPairs;
        this->spatialSize = spatialSize;
        this->channels = channels;
        this->tileSize = tileSize;
        this->constantValue = constantValue;
        this->scalingFactor = scalingFactor;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t pairsPerCore = (totalPairs + blockNum - 1) / blockNum;
        this->startPair = blockIdx * pairsPerCore;
        this->endPair = this->startPair + pairsPerCore;
        if (this->startPair > totalPairs) this->startPair = totalPairs;
        if (this->endPair > totalPairs) this->endPair = totalPairs;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, channels);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        uint32_t alignedChannels = ((channels + 7) / 8) * 8;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(biasBuf, alignedChannels * sizeof(float));
        pipe.InitBuffer(constBuf, tileSize * sizeof(float));

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopyExtParams biasCopyParams{1, (uint32_t)(channels * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> biasPadParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, biasPadParams);

        AscendC::LocalTensor<float> constLocal = constBuf.Get<float>();
        AscendC::Duplicate<float>(constLocal, this->constantValue, tileSize);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if (this->startPair >= this->endPair) return;

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();

        for (uint32_t pairIdx = this->startPair; pairIdx < this->endPair; pairIdx++) {
            uint32_t c = pairIdx % this->channels;
            float biasVal = biasLocal.GetValue(c);

            uint64_t baseOffset = (uint64_t)pairIdx * (uint64_t)this->spatialSize;

            uint32_t numTiles = (this->spatialSize + this->tileSize - 1) / this->tileSize;
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t offset = t * this->tileSize;
                uint32_t curLen = (offset + this->tileSize <= this->spatialSize)
                                    ? this->tileSize
                                    : (this->spatialSize - offset);
                CopyIn(baseOffset + offset, curLen);
                Compute(curLen, biasVal);
                CopyOut(baseOffset + offset, curLen);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len, float biasVal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> constLocal = constBuf.Get<float>();

        AscendC::Min<float>(yLocal, xLocal, constLocal, len);
        AscendC::Adds<float>(yLocal, yLocal, biasVal, len);
        AscendC::Muls<float>(yLocal, yLocal, this->scalingFactor, len);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> constBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalPairs;
    uint32_t spatialSize;
    uint32_t channels;
    uint32_t tileSize;
    uint32_t startPair;
    uint32_t endPair;
    float constantValue;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void conv2d_min_add_multiply_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMinAddMultiply op;
    op.Init(x, bias, y,
            tiling_data.totalPairs, tiling_data.spatialSize,
            tiling_data.channels, tiling_data.tileSize,
            tiling_data.constantValue, tiling_data.scalingFactor);
    op.Process();
}
