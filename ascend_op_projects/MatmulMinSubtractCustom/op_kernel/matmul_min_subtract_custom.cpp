
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelMatmulMinSubtract {
public:
    __aicore__ inline KernelMatmulMinSubtract() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR constant, GM_ADDR z,
                                uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, uint32_t totalLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outFeatures);
        constantGm.SetGlobalBuffer((__gm__ DTYPE_CONSTANT *)constant, 1);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * inFeatures * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, outFeatures * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(inQueueConstant, BUFFER_NUM, sizeof(DTYPE_CONSTANT));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileLength * BUFFER_NUM;
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
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_CONSTANT> constantLocal = inQueueConstant.AllocTensor<DTYPE_CONSTANT>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm, this->inFeatures * this->outFeatures);
        AscendC::DataCopy(biasLocal, biasGm, this->outFeatures);
        AscendC::DataCopy(constantLocal, constantGm, 1);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
        inQueueConstant.EnQue(constantLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_CONSTANT> constantLocal = inQueueConstant.DeQue<DTYPE_CONSTANT>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        
        // Matrix multiplication
        AscendC::MatMul(zLocal, xLocal, weightLocal, biasLocal, this->tileLength, this->inFeatures, this->outFeatures);
        
        // Apply min operation
        AscendC::Min(zLocal, zLocal, constantLocal, this->outFeatures);
        
        // Subtract constant
        AscendC::Sub(zLocal, zLocal, constantLocal, this->outFeatures);
        
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
        inQueueConstant.FreeTensor(constantLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias, inQueueConstant;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_CONSTANT> constantGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_min_subtract_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR constant, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulMinSubtract op;
    op.Init(x, weight, bias, constant, z, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.totalLength);
    op.Process();
}
