
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulScaleResidualAddClampLogSumExpMish {
public:
    __aicore__ inline KernelMatmulScaleResidualAddClampLogSumExpMish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t hiddenSize,
                                 float scaleFactor, float clampMin, float clampMax)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t baseRows = batchSize / blockNum;
        uint32_t extraRows = batchSize % blockNum;

        if (blockIdx < extraRows) {
            this->rowsThisCore = baseRows + 1;
            this->startRow = blockIdx * (baseRows + 1);
        } else {
            this->rowsThisCore = baseRows;
            this->startRow = extraRows * (baseRows + 1) + (blockIdx - extraRows) * baseRows;
        }

        this->hiddenSize = hiddenSize;
        this->combinedScale = 2.0f * scaleFactor;
        this->clampMin = clampMin;
        this->clampMax = clampMax;

        if (this->rowsThisCore == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x + this->startRow * hiddenSize, this->rowsThisCore * hiddenSize);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->startRow, this->rowsThisCore);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, hiddenSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, 256);
        pipe.InitBuffer(reduceBuf, 4096);
        uint32_t outBufSize = ((this->rowsThisCore * sizeof(float) + 31) / 32) * 32;
        if (outBufSize < 32) {
            outBufSize = 32;
        }
        pipe.InitBuffer(outBuf, outBufSize);
    }

    __aicore__ inline void Process()
    {
        if (rowsThisCore == 0) {
            return;
        }

        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();

        for (uint32_t i = 0; i < rowsThisCore; i++) {
            CopyIn(i);
            ComputeRow(i, outLocal);
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(rowsThisCore * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm, outLocal, copyParams);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * hiddenSize], hiddenSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeRow(uint32_t progress, AscendC::LocalTensor<float>& outLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();

        AscendC::Muls<float>(xLocal, xLocal, combinedScale, hiddenSize);
        AscendC::Mins<float>(xLocal, xLocal, clampMax, hiddenSize);
        AscendC::Maxs<float>(xLocal, xLocal, clampMin, hiddenSize);

        AscendC::ReduceMax<float>(tmp, xLocal, reduceTmp, static_cast<int32_t>(hiddenSize), false);
        float maxVal = tmp.GetValue(0);

        AscendC::Adds<float>(xLocal, xLocal, -maxVal, hiddenSize);
        AscendC::Exp<float>(xLocal, xLocal, hiddenSize);

        AscendC::ReduceSum<float, true>(tmp, xLocal, reduceTmp, static_cast<int32_t>(hiddenSize));
        float sumVal = tmp.GetValue(0);

        inQueueX.FreeTensor(xLocal);

        AscendC::Duplicate<float>(tmp, sumVal, 8);
        AscendC::Log<float>(tmp, tmp, 8);
        float logSum = tmp.GetValue(0);
        float lse = maxVal + logSum;

        AscendC::Duplicate<float>(tmp, lse, 8);
        AscendC::Exp<float>(tmp, tmp, 8);
        AscendC::Adds<float>(tmp, tmp, 1.0f, 8);
        AscendC::Log<float>(tmp, tmp, 8);
        AscendC::Tanh<float>(tmp, tmp, 8);
        float tanhSoftplus = tmp.GetValue(0);

        float mishVal = lse * tanhSoftplus;
        float result = lse * mishVal;

        outLocal.SetValue(progress, result);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t hiddenSize;
    uint32_t rowsThisCore;
    uint32_t startRow;
    float combinedScale;
    float clampMin;
    float clampMax;
};

extern "C" __global__ __aicore__ void matmul_scale_residual_add_clamp_log_sum_exp_mish_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulScaleResidualAddClampLogSumExpMish op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.hiddenSize,
            tiling_data.scaleFactor, tiling_data.clampMin, tiling_data.clampMax);
    op.Process();
}
