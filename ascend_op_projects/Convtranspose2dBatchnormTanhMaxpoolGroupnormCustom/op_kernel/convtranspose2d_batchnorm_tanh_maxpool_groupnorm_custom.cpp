
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTanhFused {
public:
    __aicore__ inline KernelTanhFused() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t basePerBlock = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;

        if (blockIdx < remainder) {
            this->blockLength = basePerBlock + 1;
            this->blockOffset = blockIdx * this->blockLength;
        } else {
            this->blockLength = basePerBlock;
            this->blockOffset = remainder * (basePerBlock + 1) + (blockIdx - remainder) * basePerBlock;
        }

        if (this->blockLength == 0) {
            this->tileNum = 0;
            this->tileLength = 0;
            this->tailLength = 0;
            return;
        }

        this->tileNum = tileNum;
        uint32_t totalTiles = this->tileNum * BUFFER_NUM;
        this->tileLength = this->blockLength / totalTiles;

        uint32_t alignElems = 8;
        if (this->tileLength < alignElems) {
            this->tileLength = alignElems;
        } else {
            this->tileLength = (this->tileLength / alignElems) * alignElems;
        }

        if (this->tileLength == 0) {
            this->tileLength = this->blockLength;
        }

        this->fullLoopCount = this->blockLength / this->tileLength;
        this->tailLength = this->blockLength - this->fullLoopCount * this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockOffset, this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;

        for (uint32_t i = 0; i < this->fullLoopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(this->tileLength);
            CopyOut(i, this->tileLength);
        }

        if (this->tailLength > 0) {
            CopyInTail(this->fullLoopCount, this->tailLength);
            Compute(this->tailLength);
            CopyOutTail(this->fullLoopCount, this->tailLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInTail(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DTYPE_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DTYPE_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::Tanh(yLocal, xLocal, len);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutTail(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(DTYPE_Y)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t blockOffset;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t fullLoopCount;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void convtranspose2d_batchnorm_tanh_maxpool_groupnorm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTanhFused op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
