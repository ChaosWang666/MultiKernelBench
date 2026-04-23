
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                  uint32_t totalFeatureMaps, uint32_t featureMapSize,
                                  uint32_t channels, uint32_t tileLength)
    {
        this->totalFeatureMaps = totalFeatureMaps;
        this->featureMapSize = featureMapSize;
        this->channels = channels;
        this->tileLength = tileLength;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t fmPerBlock = totalFeatureMaps / blockNum;
        uint32_t fmRemainder = totalFeatureMaps % blockNum;

        if (blockIdx < fmRemainder) {
            this->startFm = blockIdx * (fmPerBlock + 1);
            this->numFm = fmPerBlock + 1;
        } else {
            this->startFm = blockIdx * fmPerBlock + fmRemainder;
            this->numFm = fmPerBlock;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)totalFeatureMaps * featureMapSize);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, channels);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)totalFeatureMaps * featureMapSize);

        uint32_t channelsAlign = ((channels + 7) / 8) * 8;
        this->channelsAlign = channelsAlign;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(biasQueue, 1, channelsAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->numFm == 0) return;

        auto biasAlloc = biasQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams biasCopyParams{1, (uint32_t)(this->channels * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> biasPadParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(biasAlloc, biasGm, biasCopyParams, biasPadParams);
        biasQueue.EnQue(biasAlloc);
        auto biasIn = biasQueue.DeQue<float>();

        for (uint32_t fm = 0; fm < this->numFm; fm++) {
            uint32_t globalFm = this->startFm + fm;
            uint32_t channel = globalFm % this->channels;
            float biasVal = biasIn.GetValue(channel);
            float biasPlus1 = biasVal + 1.0f;

            uint64_t fmOffset = (uint64_t)globalFm * this->featureMapSize;
            uint32_t processed = 0;
            while (processed < this->featureMapSize) {
                uint32_t remaining = this->featureMapSize - processed;
                uint32_t thisTile = (remaining < this->tileLength) ? remaining : this->tileLength;

                CopyIn(fmOffset + processed, thisTile);
                Compute(thisTile, biasPlus1);
                CopyOut(fmOffset + processed, thisTile);

                processed += thisTile;
            }
        }

        biasQueue.FreeTensor(biasIn);
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length)
    {
        auto xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(length * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length, float biasPlus1)
    {
        auto xLocal = inQueueX.DeQue<float>();
        auto yLocal = outQueueY.AllocTensor<float>();

        AscendC::Muls<float>(yLocal, xLocal, 2.0f, length);
        AscendC::Adds<float>(yLocal, yLocal, biasPlus1, length);
        AscendC::Mul<float>(yLocal, xLocal, yLocal, length);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length)
    {
        auto yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)(length * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalFeatureMaps;
    uint32_t featureMapSize;
    uint32_t channels;
    uint32_t channelsAlign;
    uint32_t tileLength;
    uint32_t startFm;
    uint32_t numFm;
};

extern "C" __global__ __aicore__ void conv_transpose3d_sum_residual_add_multiply_residual_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, y, tiling_data.totalFeatureMaps, tiling_data.featureMapSize,
            tiling_data.channels, tiling_data.tileLength);
    op.Process();
}
