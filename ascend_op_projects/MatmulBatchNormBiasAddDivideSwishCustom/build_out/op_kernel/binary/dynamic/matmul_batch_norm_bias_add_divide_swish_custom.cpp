
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulBatchNormBiasAddDivideSwish {
public:
    __aicore__ inline KernelMatmulBatchNormBiasAddDivideSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t totalLength, uint32_t tileNum, float divideValue)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->invDivideValue = 1.0f / divideValue;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, 1);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(biasBuf, 32);

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopyExtParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = (uint32_t)sizeof(float);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        biasCopyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> biasPadParams;
        biasPadParams.isPad = false;
        biasPadParams.leftPadding = 0;
        biasPadParams.rightPadding = 0;
        biasPadParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, biasPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        this->biasValue = biasLocal.GetValue(0);
        this->scaledBias = this->biasValue * this->invDivideValue;
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        AscendC::Muls(tmpLocal, xLocal, this->invDivideValue, this->tileLength);
        AscendC::Adds(tmpLocal, tmpLocal, this->scaledBias, this->tileLength);

        AscendC::Muls(xLocal, tmpLocal, -1.0f, this->tileLength);
        AscendC::Exp(xLocal, xLocal, this->tileLength);
        AscendC::Adds(xLocal, xLocal, 1.0f, this->tileLength);
        AscendC::Div(yLocal, tmpLocal, xLocal, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float invDivideValue;
    float biasValue;
    float scaledBias;
};

extern "C" __global__ __aicore__ void matmul_batch_norm_bias_add_divide_swish_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulBatchNormBiasAddDivideSwish op;
    op.Init(x, bias, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.divideValue);
    op.Process();
}
