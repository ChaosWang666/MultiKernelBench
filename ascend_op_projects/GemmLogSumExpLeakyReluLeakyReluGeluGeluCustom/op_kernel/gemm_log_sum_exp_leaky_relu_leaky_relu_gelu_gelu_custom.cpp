
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmLogSumExpLeakyReluLeakyReluGeluGelu {
public:
    __aicore__ inline KernelGemmLogSumExpLeakyReluLeakyReluGeluGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, bool hasBias)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->hasBias = hasBias;
        this->blockLength = batchSize * outFeatures / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inFeatures * outFeatures);
        if (hasBias) {
            biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outFeatures);
        }
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, batchSize * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        // Gemm
        Gemm();
        // LogSumExp
        LogSumExp();
        // LeakyReLU
        LeakyRelu(0.01f);
        // LeakyReLU
        LeakyRelu(0.01f);
        // GELU
        Gelu();
        // GELU
        Gelu();
    }

private:
    __aicore__ inline void Gemm()
    {
        int32_t loopCount = this->batchSize * this->outFeatures / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

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
        AscendC::MatMul(zLocal, xLocal, weightLocal, this->tileLength);
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

    __aicore__ inline void LogSumExp()
    {
        // Placeholder for LogSumExp implementation
    }

    __aicore__ inline void LeakyRelu(float negativeSlope)
    {
        // Placeholder for LeakyReLU implementation
    }

    __aicore__ inline void Gelu()
    {
        // Placeholder for GELU implementation
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    bool hasBias;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmLogSumExpLeakyReluLeakyReluGeluGelu op;
    op.Init(x, weight, bias, z, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.hasBias);
    op.Process();
}
