
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelBmmInstanceNormSumResidualAddMultiply {
public:
    __aicore__ inline KernelBmmInstanceNormSumResidualAddMultiply() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, 
                                uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outFeatures);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * inFeatures * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
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
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], this->tileLength * this->inFeatures);
        AscendC::DataCopy(biasLocal, biasGm[0], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        
        // BMM operation
        AscendC::LocalTensor<DTYPE_Z> tempLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::MatMul(tempLocal, xLocal, weightLocal, this->tileLength, this->inFeatures, this->outFeatures);
        
        // Instance Norm
        AscendC::LocalTensor<DTYPE_Z> normLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::InstanceNorm(normLocal, tempLocal, biasLocal, this->tileLength, this->outFeatures);
        
        // Sum
        AscendC::LocalTensor<DTYPE_Z> sumLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(sumLocal, normLocal, yLocal, this->tileLength);
        
        // Residual Add
        AscendC::LocalTensor<DTYPE_Z> residualLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(residualLocal, sumLocal, yLocal, this->tileLength);
        
        // Multiply
        AscendC::Mul(zLocal, residualLocal, yLocal, this->tileLength);
        
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
};

extern "C" __global__ __aicore__ void bmm_instance_norm_sum_residual_add_multiply_custom(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBmmInstanceNormSumResidualAddMultiply op;
    op.Init(x, y, weight, bias, z, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
