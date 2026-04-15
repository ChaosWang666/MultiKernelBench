
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelMatmulScalingResidualAdd {
public:
    __aicore__ inline KernelMatmulScalingResidualAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR z, uint32_t batch, uint32_t inFeatures, uint32_t outFeatures, float scalingFactor)
    {
        this->batch = batch;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->scalingFactor = scalingFactor;
        this->blockLength = inFeatures * outFeatures / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batch * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inFeatures * outFeatures);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, batch * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
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
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::MatMul(zLocal, xLocal, weightLocal, this->tileLength, this->tileLength, this->tileLength, false, false);
        AscendC::Mul(zLocal, zLocal, this->scalingFactor, this->tileLength);
        AscendC::Add(zLocal, zLocal, xLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batch;
    uint32_t inFeatures;
    uint32_t outFeatures;
    float scalingFactor;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_scaling_residual_add_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulScalingResidualAdd op;
    op.Init(x, weight, z, tiling_data.batch, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.scalingFactor);
    op.Process();
}
