
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmScaleBatchNorm {
public:
    __aicore__ inline KernelGemmScaleBatchNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                GM_ADDR mean, GM_ADDR variance, GM_ADDR y,
                                uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->tileNum = tileNum;
        this->blockLength = outFeatures / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, outFeatures);
        meanGm.SetGlobalBuffer((__gm__ float *)mean, outFeatures);
        varianceGm.SetGlobalBuffer((__gm__ float *)variance, outFeatures);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * outFeatures);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueScale, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueMean, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueVariance, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        AscendC::LocalTensor<float> scaleLocal = inQueueScale.AllocTensor<float>();
        AscendC::LocalTensor<float> meanLocal = inQueueMean.AllocTensor<float>();
        AscendC::LocalTensor<float> varianceLocal = inQueueVariance.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(scaleLocal, scaleGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(meanLocal, meanGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(varianceLocal, varianceGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
        inQueueScale.EnQue(scaleLocal);
        inQueueMean.EnQue(meanLocal);
        inQueueVariance.EnQue(varianceLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> scaleLocal = inQueueScale.DeQue<float>();
        AscendC::LocalTensor<float> meanLocal = inQueueMean.DeQue<float>();
        AscendC::LocalTensor<float> varianceLocal = inQueueVariance.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // GEMM operation
        AscendC::MatMul(yLocal, xLocal, weightLocal, this->tileLength, this->tileLength, this->tileLength);

        // Scale operation
        AscendC::Mul(yLocal, yLocal, scaleLocal, this->tileLength);

        // BatchNorm operation
        AscendC::Sub(yLocal, yLocal, meanLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, varianceLocal, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
        inQueueScale.FreeTensor(scaleLocal);
        inQueueMean.FreeTensor(meanLocal);
        inQueueVariance.FreeTensor(varianceLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias, inQueueScale, inQueueMean, inQueueVariance;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> meanGm;
    AscendC::GlobalTensor<float> varianceGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_scale_batch_norm_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
    GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmScaleBatchNorm op;
    op.Init(x, weight, bias, scale, mean, variance, y,
            tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.tileNum);
    op.Process();
}
