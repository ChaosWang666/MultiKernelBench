
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSigmoidSum {
public:
    __aicore__ inline KernelSigmoidSum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchElements,
                                 uint32_t tileLength)
    {
        this->batchElements = batchElements;
        this->tileLength = tileLength;
        uint32_t batchIdx = AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ float*)x + batchIdx * batchElements, batchElements);
        yGm.SetGlobalBuffer((__gm__ float*)y + batchIdx, 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, 32);
        pipe.InitBuffer(reduceBuf, 8192);
        pipe.InitBuffer(resultBuf, 32);
    }

    __aicore__ inline void Process()
    {
        uint32_t tileNum = (batchElements + tileLength - 1) / tileLength;
        float totalSum = 0.0f;

        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t curTileLen = (i == tileNum - 1)
                ? (batchElements - i * tileLength)
                : tileLength;
            CopyIn(i, curTileLen);
            totalSum += Compute(curTileLen);
        }

        WriteResult(totalSum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curTileLen)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curTileLen * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline float Compute(uint32_t curTileLen)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();

        // sigmoid(x) = 1 / (1 + exp(-x))
        AscendC::Muls<float>(xLocal, xLocal, -1.0f, static_cast<int32_t>(curTileLen));
        AscendC::Exp<float>(xLocal, xLocal, static_cast<int32_t>(curTileLen));
        AscendC::Adds<float>(xLocal, xLocal, 1.0f, static_cast<int32_t>(curTileLen));
        AscendC::Reciprocal<float>(xLocal, xLocal, static_cast<int32_t>(curTileLen));

        AscendC::ReduceSum<float, true>(resultLocal, xLocal, reduceTmp, static_cast<int32_t>(curTileLen));
        float partialSum = resultLocal.GetValue(0);

        inQueueX.FreeTensor(xLocal);
        return partialSum;
    }

    __aicore__ inline void WriteResult(float totalSum)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        zLocal.SetValue(0, totalSum);
        outQueueZ.EnQue(zLocal);

        AscendC::LocalTensor<float> zOut = outQueueZ.DeQue<float>();
        AscendC::DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm, zOut, copyOutParams);
        outQueueZ.FreeTensor(zOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchElements;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_avg_pool_sigmoid_sum_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoidSum op;
    op.Init(x, y, tiling_data.batchElements, tiling_data.tileLength);
    op.Process();
}
