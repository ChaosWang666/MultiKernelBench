
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGeluRelu {
public:
    __aicore__ inline KernelGeluRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(DTYPE_X));
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
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf.Get<DTYPE_X>();

        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // Step 1: tmp = x / sqrt(2) = x * 0.7071067811865476
        AscendC::Muls(tmp, xLocal, (DTYPE_X)0.7071067811865476f, this->tileLength);
        // Step 2: tmp = erf(x / sqrt(2))
        AscendC::Erf(tmp, tmp, this->tileLength);
        // Step 3: tmp = 1 + erf(x / sqrt(2))
        AscendC::Adds(tmp, tmp, (DTYPE_X)1.0f, this->tileLength);
        // Step 4: tmp = x * (1 + erf(x / sqrt(2)))
        AscendC::Mul(tmp, xLocal, tmp, this->tileLength);
        // Step 5: tmp = 0.5 * x * (1 + erf(x / sqrt(2))) = GELU(x)
        AscendC::Muls(tmp, tmp, (DTYPE_X)0.5f, this->tileLength);

        // ReLU: y = max(0, tmp) using identity max(0,x) = (x + |x|) / 2
        AscendC::Abs(yLocal, tmp, this->tileLength);
        AscendC::Add(yLocal, yLocal, tmp, this->tileLength);
        AscendC::Muls(yLocal, yLocal, (DTYPE_X)0.5f, this->tileLength);

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
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_batch_norm_gelu_group_norm_mean_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGeluRelu op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
