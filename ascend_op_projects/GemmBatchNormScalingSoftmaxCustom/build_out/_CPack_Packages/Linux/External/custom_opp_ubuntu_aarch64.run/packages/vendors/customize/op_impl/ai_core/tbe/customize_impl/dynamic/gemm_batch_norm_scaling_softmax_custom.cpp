
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmBatchNormScalingSoftmax {
public:
    __aicore__ inline KernelGemmBatchNormScalingSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling, uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, float bnEps, float bnMomentum, uint32_t scaleShape0)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->bnEps = bnEps;
        this->bnMomentum = bnMomentum;
        this->scaleShape0 = scaleShape0;
        this->blockLength = outFeatures;
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * inFeatures);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            uint32_t offset = batch * outFeatures;
            xGm.SetGlobalBuffer((__gm__ float *)xGm.GetBaseAddr() + offset, outFeatures);
            yGm.SetGlobalBuffer((__gm__ float *)yGm.GetBaseAddr() + offset, outFeatures);
            int32_t loopCount = this->tileNum * BUFFER_NUM;
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // Simulate Gemm + BatchNorm + Scaling + Softmax operations
        for (int32_t i = 0; i < this->tileLength; i++) {
            float val = xLocal[i];
            // BatchNorm
            val = val; // Placeholder for actual BN computation
            // Scaling
            val = val * 1.0f; // Placeholder for actual scaling
            // Softmax
            val = val; // Placeholder for actual softmax
            yLocal[i] = val;
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    float bnEps;
    float bnMomentum;
    uint32_t scaleShape0;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_batch_norm_scaling_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmBatchNormScalingSoftmax op;
    op.Init(x, y, workspace, tiling, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.bnEps, tiling_data.bnMomentum, tiling_data.scaleShape0);
    op.Process();
}
