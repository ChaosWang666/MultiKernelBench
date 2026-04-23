
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t totalLength, uint32_t tileLength)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t perBlock = (totalLength + blockNum - 1) / blockNum;
        perBlock = ((perBlock + 7) / 8) * 8;

        uint32_t start = blockIdx * perBlock;
        uint32_t end = start + perBlock;
        if (end > totalLength) end = totalLength;

        if (start >= totalLength) {
            this->myLength = 0;
        } else {
            this->myLength = end - start;
        }

        this->tileLength = tileLength;

        if (this->myLength > 0) {
            xGm.SetGlobalBuffer((__gm__ float*)x + start, this->myLength);
            yGm.SetGlobalBuffer((__gm__ float*)y + start, this->myLength);
        }
        biasGm.SetGlobalBuffer((__gm__ float*)bias, 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(biasBuf, 32);

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopyExtParams biasCopy;
        biasCopy.blockCount = 1;
        biasCopy.blockLen = sizeof(float);
        biasCopy.srcStride = 0;
        biasCopy.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> biasPad;
        biasPad.isPad = false;
        biasPad.leftPadding = 0;
        biasPad.rightPadding = 0;
        biasPad.paddingValue = 0.0f;
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopy, biasPad);
        AscendC::PipeBarrier<PIPE_ALL>();
        this->biasVal = biasLocal.GetValue(0);
    }

    __aicore__ inline void Process()
    {
        if (this->myLength == 0) return;

        uint32_t processed = 0;
        while (processed < this->myLength) {
            uint32_t curTile = this->tileLength;
            if (processed + curTile > this->myLength) {
                curTile = this->myLength - processed;
            }
            CopyIn(processed, curTile);
            Compute(curTile);
            CopyOut(processed, curTile);
            processed += curTile;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curTile)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curTile * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curTile)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();

        // hardswish(x) = x * sigmoid(x+3) / 6 = x / (6 + 6*exp(-x-3))
        AscendC::Muls<float>(tmp, xLocal, -1.0f, curTile);
        AscendC::Adds<float>(tmp, tmp, -3.0f, curTile);
        AscendC::Exp<float>(tmp, tmp, curTile);
        AscendC::Muls<float>(tmp, tmp, 6.0f, curTile);
        AscendC::Adds<float>(tmp, tmp, 6.0f, curTile);
        AscendC::Div<float>(yLocal, xLocal, tmp, curTile);

        // subtract bias
        AscendC::Adds<float>(yLocal, yLocal, -this->biasVal, curTile);

        // clamp to [-1, 1]
        AscendC::Duplicate<float>(tmp, -1.0f, curTile);
        AscendC::Max<float>(yLocal, yLocal, tmp, curTile);
        AscendC::Duplicate<float>(tmp, 1.0f, curTile);
        AscendC::Min<float>(yLocal, yLocal, tmp, curTile);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curTile)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = curTile * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t myLength;
    uint32_t tileLength;
    float biasVal;
};

extern "C" __global__ __aicore__ void conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, y, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
