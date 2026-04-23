
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelLseOp {
public:
    __aicore__ inline KernelLseOp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t batchSize, uint32_t channels)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->scale = 10.0f;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t batchesPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startBatch = blockIdx * batchesPerBlock;
        uint32_t endB = this->startBatch + batchesPerBlock;
        this->endBatch = (endB > batchSize) ? batchSize : endB;

        if (this->startBatch >= batchSize) {
            this->hasWork = false;
            return;
        }
        this->hasWork = true;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, channels);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize);

        this->alignedChannels = ((channels + 7) / 8) * 8;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedChannels * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, alignedChannels * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
        pipe.InitBuffer(reduceTmpBuf, 4 * 1024);
        pipe.InitBuffer(scalarBuf, 64);
    }

    __aicore__ inline void Process()
    {
        if (!hasWork) return;

        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        AscendC::DataCopyExtParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = channels * sizeof(float);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, padParams);
        inQueueBias.EnQue(biasLocal);
        AscendC::LocalTensor<float> biasIn = inQueueBias.DeQue<float>();

        for (uint32_t b = startBatch; b < endBatch; b++) {
            ProcessBatch(b, biasIn);
        }

        inQueueBias.FreeTensor(biasIn);
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx, AscendC::LocalTensor<float>& biasIn)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = channels * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(xLocal, xGm[batchIdx * channels], copyInParams, padParams);
        inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<float> x = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> scalarTmp = scalarBuf.Get<float>();

        AscendC::Add<float>(x, x, biasIn, channels);

        AscendC::ReduceMax<float>(scalarTmp, x, reduceTmp, (int32_t)channels, false);
        float maxVal = scalarTmp.GetValue(0);

        AscendC::Adds<float>(x, x, -maxVal, channels);
        AscendC::Exp<float>(x, x, channels);

        AscendC::ReduceSum<float, true>(scalarTmp, x, reduceTmp, (int32_t)channels);
        float sumExp = scalarTmp.GetValue(0);

        AscendC::Duplicate<float>(scalarTmp, sumExp, 8);
        AscendC::Ln<float>(scalarTmp, scalarTmp, 8);
        float logSum = scalarTmp.GetValue(0);

        float result = (maxVal + logSum) * scale;

        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        yLocal.SetValue(0, result);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(x);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = sizeof(float);
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[batchIdx], yOut, outParams);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t alignedChannels;
    uint32_t startBatch;
    uint32_t endBatch;
    float scale;
    bool hasWork;
};

extern "C" __global__ __aicore__ void convtranspose2d_globalavgpool_biasadd_logsumexp_sum_multiply_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLseOp op;
    op.Init(x, bias, y, tiling_data.batchSize, tiling_data.channels);
    op.Process();
}
