
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelMambaReturnY {
public:
    __aicore__ inline KernelMambaReturnY() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        aGm.SetGlobalBuffer((__gm__ DTYPE_A *)a + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        bGm.SetGlobalBuffer((__gm__ DTYPE_B *)b + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        cGm.SetGlobalBuffer((__gm__ DTYPE_C *)c + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileLength * sizeof(DTYPE_A));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B));
        pipe.InitBuffer(inQueueC, BUFFER_NUM, this->tileLength * sizeof(DTYPE_C));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.AllocTensor<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_B> bLocal = inQueueB.AllocTensor<DTYPE_B>();
        AscendC::LocalTensor<DTYPE_C> cLocal = inQueueC.AllocTensor<DTYPE_C>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(aLocal, aGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(bLocal, bGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(cLocal, cGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
        inQueueC.EnQue(cLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_A> aLocal = inQueueA.DeQue<DTYPE_A>();
        AscendC::LocalTensor<DTYPE_B> bLocal = inQueueB.DeQue<DTYPE_B>();
        AscendC::LocalTensor<DTYPE_C> cLocal = inQueueC.DeQue<DTYPE_C>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        // Placeholder for actual computation logic
        AscendC::Add(zLocal, xLocal, aLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueC.FreeTensor(cLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueA, inQueueB, inQueueC;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_A> aGm;
    AscendC::GlobalTensor<DTYPE_B> bGm;
    AscendC::GlobalTensor<DTYPE_C> cGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void mamba_return_y_custom(GM_ADDR x, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMambaReturnY op;
    op.Init(x, a, b, c, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
