
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmSwishDivideClampTanhClamp {
public:
    __aicore__ inline KernelGemmSwishDivideClampTanhClamp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, uint32_t batch, uint32_t inFeatures, uint32_t outFeatures, bool hasBias)
    {
        this->batch = batch;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->hasBias = hasBias;
        this->blockLength = inFeatures;
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batch * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inFeatures * outFeatures);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        }
        yGm.SetGlobalBuffer((__gm__ float *)y, batch * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * outFeatures * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outFeatures * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < batch; i++) {
            Gemm(i);
        }
    }

private:
    __aicore__ inline void Gemm(uint32_t batchId)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::GlobalTensor<float> xBatch = xGm[batchId * inFeatures];
        AscendC::GlobalTensor<float> yBatch = yGm[batchId * outFeatures];

        // Perform GEMM operation
        AscendC::MatMul(yLocal, xBatch, weightGm, biasGm, false, false, outFeatures, inFeatures, 1, hasBias);
        
        // Apply Swish activation: x * sigmoid(x)
        AscendC::Sigmoid(yLocal, yLocal, outFeatures);
        AscendC::Mul(yLocal, yLocal, xBatch, outFeatures);
        
        // Divide by 2.0
        AscendC::ScalarDiv(yLocal, yLocal, 2.0f, outFeatures);
        
        // Clamp between -1.0 and 1.0
        AscendC::Clip(yLocal, yLocal, -1.0f, 1.0f, outFeatures);
        
        // Tanh activation
        AscendC::Tanh(yLocal, yLocal, outFeatures);
        
        // Clamp again between -1.0 and 1.0
        AscendC::Clip(yLocal, yLocal, -1.0f, 1.0f, outFeatures);
        
        AscendC::DataCopy(yBatch, yLocal, outFeatures);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t inFeatures;
    uint32_t outFeatures;
    bool hasBias;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_swish_divide_clamp_tanh_clamp_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmSwishDivideClampTanhClamp op;
    op.Init(x, weight, bias, y, tiling_data.batch, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.hasBias);
    op.Process();
}
