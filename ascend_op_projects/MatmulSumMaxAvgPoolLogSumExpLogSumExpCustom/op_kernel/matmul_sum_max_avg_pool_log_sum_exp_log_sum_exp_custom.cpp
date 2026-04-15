
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp {
public:
    __aicore__ inline KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, 
                                uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->totalElements = batchSize * outFeatures;
        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->tileNum = 1024;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        outputGm.SetGlobalBuffer((__gm__ float *)output, this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Step 1: Matrix multiplication
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        // Step 2: Summation
        // Step 3: Max
        // Step 4: Average pooling
        // Step 5: LogSumExp
        // Step 6: LogSumExp
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueueOutput.AllocTensor<float>();
        // Perform matrix multiplication
        AscendC::MatMul(outputLocal, xLocal, weightLocal, biasLocal, this->tileLength, this->tileLength, this->tileLength);
        outQueueOutput.EnQue<float>(outputLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outputLocal = outQueueOutput.DeQue<float>();
        AscendC::DataCopy(outputGm[progress * this->tileLength], outputLocal, this->tileLength);
        outQueueOutput.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOutput;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t totalElements;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_sum_max_avg_pool_log_sum_exp_log_sum_exp_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulSumMaxAvgPoolLogSumExpLogSumExp op;
    op.Init(x, weight, bias, output, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures);
    op.Process();
}
