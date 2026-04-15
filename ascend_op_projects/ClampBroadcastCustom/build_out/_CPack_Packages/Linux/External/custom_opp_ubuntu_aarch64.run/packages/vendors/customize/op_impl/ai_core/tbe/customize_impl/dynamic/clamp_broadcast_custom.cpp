
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelClampBroadcast {
public:
    __aicore__ inline KernelClampBroadcast() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR min_val, GM_ADDR max_val, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        minGm.SetGlobalBuffer((__gm__ DTYPE_MIN *)min_val, this->blockLength);
        maxGm.SetGlobalBuffer((__gm__ DTYPE_MAX *)max_val, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueMin, BUFFER_NUM, this->tileLength * sizeof(DTYPE_MIN));
        pipe.InitBuffer(inQueueMax, BUFFER_NUM, this->tileLength * sizeof(DTYPE_MAX));
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
        AscendC::LocalTensor<DTYPE_MIN> minLocal = inQueueMin.AllocTensor<DTYPE_MIN>();
        AscendC::LocalTensor<DTYPE_MAX> maxLocal = inQueueMax.AllocTensor<DTYPE_MAX>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(minLocal, minGm[0], this->tileLength);
        AscendC::DataCopy(maxLocal, maxGm[0], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueMin.EnQue(minLocal);
        inQueueMax.EnQue(maxLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_MIN> minLocal = inQueueMin.DeQue<DTYPE_MIN>();
        AscendC::LocalTensor<DTYPE_MAX> maxLocal = inQueueMax.DeQue<DTYPE_MAX>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Clamp(zLocal, xLocal, minLocal, maxLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMin.FreeTensor(minLocal);
        inQueueMax.FreeTensor(maxLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueMin, inQueueMax;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_MIN> minGm;
    AscendC::GlobalTensor<DTYPE_MAX> maxGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void clamp_broadcast_custom(GM_ADDR x, GM_ADDR min_val, GM_ADDR max_val, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelClampBroadcast op;
    op.Init(x, min_val, max_val, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
