
#include "kernel_operator.h"

constexpr float EPS_VAL = 1e-5f;
constexpr float NEG_SLOPE_VAL = 0.01f;

class KernelMGNLRS {
public:
    __aicore__ inline KernelMGNLRS() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                uint32_t batchSize, uint32_t hiddenSize,
                                uint32_t numGroups, uint32_t channelsPerGroup)
    {
        this->batchSize = batchSize;
        this->hiddenSize = hiddenSize;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->invN = 1.0f / (float)channelsPerGroup;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t rowsPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startRow = blockIdx * rowsPerBlock;
        uint32_t tmpEnd = this->startRow + rowsPerBlock;
        this->endRow = (tmpEnd > batchSize) ? batchSize : tmpEnd;
        this->rowsThisBlock = (this->endRow > this->startRow) ? (this->endRow - this->startRow) : 0;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * hiddenSize);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, hiddenSize);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, hiddenSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * hiddenSize);

        pipe.InitBuffer(inQueueX, 1, hiddenSize * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, hiddenSize * sizeof(float));
        pipe.InitBuffer(gammaBuf, hiddenSize * sizeof(float));
        pipe.InitBuffer(betaBuf, hiddenSize * sizeof(float));
        pipe.InitBuffer(sumBuf, numGroups * sizeof(float) + 32);
        pipe.InitBuffer(sumSqBuf, numGroups * sizeof(float) + 32);
        pipe.InitBuffer(meanSqBuf, numGroups * sizeof(float) + 32);
        pipe.InitBuffer(reduceTmpBuf, 16 * 1024);
    }

    __aicore__ inline void Process()
    {
        if (rowsThisBlock == 0) return;

        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> betaLocal = betaBuf.Get<float>();

        AscendC::DataCopyExtParams cpParams;
        cpParams.blockCount = 1;
        cpParams.blockLen = hiddenSize * sizeof(float);
        cpParams.srcStride = 0;
        cpParams.dstStride = 0;
        cpParams.rsv = 0;

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;

        AscendC::DataCopyPad(gammaLocal, gammaGm, cpParams, padParams);
        AscendC::DataCopyPad(betaLocal, betaGm, cpParams, padParams);

        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t rowIdx = 0; rowIdx < rowsThisBlock; rowIdx++) {
            uint32_t row = startRow + rowIdx;
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        AscendC::DataCopyExtParams cpIn;
        cpIn.blockCount = 1;
        cpIn.blockLen = hiddenSize * sizeof(float);
        cpIn.srcStride = 0;
        cpIn.dstStride = 0;
        cpIn.rsv = 0;

        AscendC::DataCopyPadExtParams<float> padIn;
        padIn.isPad = false;
        padIn.leftPadding = 0;
        padIn.rightPadding = 0;
        padIn.paddingValue = 0.0f;

        AscendC::DataCopyPad(xLocal, xGm[row * hiddenSize], cpIn, padIn);
        inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<float> x = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> y = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> sumL = sumBuf.Get<float>();
        AscendC::LocalTensor<float> sumSqL = sumSqBuf.Get<float>();
        AscendC::LocalTensor<float> meanSq = meanSqBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> reduceTmp = reduceTmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> betaLocal = betaBuf.Get<float>();

        uint32_t srcShape[2] = {numGroups, channelsPerGroup};

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumL, x, reduceTmp, srcShape, true);

        AscendC::Mul(y, x, x, (int32_t)hiddenSize);

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumSqL, y, reduceTmp, srcShape, true);

        AscendC::Muls(sumL, sumL, invN, (int32_t)numGroups);
        AscendC::Muls(sumSqL, sumSqL, invN, (int32_t)numGroups);
        AscendC::Mul(meanSq, sumL, sumL, (int32_t)numGroups);
        AscendC::Sub(sumSqL, sumSqL, meanSq, (int32_t)numGroups);
        AscendC::Adds(sumSqL, sumSqL, EPS_VAL, (int32_t)numGroups);
        AscendC::Sqrt(sumSqL, sumSqL, (int32_t)numGroups);

        for (uint32_t g = 0; g < numGroups; g++) {
            uint32_t off = g * channelsPerGroup;
            float mean = sumL.GetValue(g);
            float stddev = sumSqL.GetValue(g);
            float rstd = 1.0f / stddev;

            AscendC::Adds(y[off], x[off], -mean, (int32_t)channelsPerGroup);
            AscendC::Muls(y[off], y[off], rstd, (int32_t)channelsPerGroup);
        }

        AscendC::Mul(y, y, gammaLocal, (int32_t)hiddenSize);
        AscendC::Add(y, y, betaLocal, (int32_t)hiddenSize);

        AscendC::Muls(x, y, NEG_SLOPE_VAL, (int32_t)hiddenSize);
        AscendC::Max(y, y, x, (int32_t)hiddenSize);

        AscendC::Muls(y, y, 2.0f, (int32_t)hiddenSize);

        outQueueY.EnQue<float>(y);
        inQueueX.FreeTensor(x);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();

        AscendC::DataCopyExtParams cpOut;
        cpOut.blockCount = 1;
        cpOut.blockLen = hiddenSize * sizeof(float);
        cpOut.srcStride = 0;
        cpOut.dstStride = 0;
        cpOut.rsv = 0;

        AscendC::DataCopyPad(yGm[row * hiddenSize], yOut, cpOut);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumSqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> meanSqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> gammaGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t batchSize;
    uint32_t hiddenSize;
    uint32_t numGroups;
    uint32_t channelsPerGroup;
    float invN;
    uint32_t startRow;
    uint32_t endRow;
    uint32_t rowsThisBlock;
};

extern "C" __global__ __aicore__ void matmul_group_norm_leaky_relu_sum_custom(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMGNLRS op;
    op.Init(x, gamma, beta, y,
            tiling_data.batchSize, tiling_data.hiddenSize,
            tiling_data.numGroups, tiling_data.channelsPerGroup);
    op.Process();
}
