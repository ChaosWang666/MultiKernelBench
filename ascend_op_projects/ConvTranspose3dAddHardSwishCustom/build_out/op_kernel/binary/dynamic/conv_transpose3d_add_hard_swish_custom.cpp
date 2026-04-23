
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConvTranspose3dAddHardSwish {
public:
    __aicore__ inline KernelConvTranspose3dAddHardSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(sixBuf, this->tileLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_Z> tmpLocal = tmpBuf.Get<DTYPE_Z>();
        AscendC::LocalTensor<DTYPE_Z> sixLocal = sixBuf.Get<DTYPE_Z>();

        // t = x + y  -> zLocal
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        // tmp = t + 3
        AscendC::Adds(tmpLocal, zLocal, (DTYPE_Z)3.0f, this->tileLength);
        // tmp = max(tmp, 0)  (ReLU)
        AscendC::Relu(tmpLocal, tmpLocal, this->tileLength);
        // sixLocal = 6.0
        AscendC::Duplicate(sixLocal, (DTYPE_Z)6.0f, this->tileLength);
        // tmp = min(tmp, 6)  -> ReLU6(t+3)
        AscendC::Min(tmpLocal, tmpLocal, sixLocal, this->tileLength);
        // tmp = tmp * (1/6)  -> hardswish(t) / t
        AscendC::Muls(tmpLocal, tmpLocal, (DTYPE_Z)(1.0f / 6.0f), this->tileLength);
        // tmp = t * tmp   -> hardswish(t)
        AscendC::Mul(tmpLocal, zLocal, tmpLocal, this->tileLength);
        // z = t * hardswish(t)
        AscendC::Mul(zLocal, zLocal, tmpLocal, this->tileLength);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sixBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_add_hard_swish_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dAddHardSwish op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
