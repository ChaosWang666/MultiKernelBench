
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelMatmulSwishSumGroupNorm {
public:
    __aicore__ inline KernelMatmulSwishSumGroupNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupNormWeight, GM_ADDR groupNormBias,
                                GM_ADDR output, uint32_t batch, uint32_t inFeatures, uint32_t outFeatures, uint32_t numGroups, uint32_t totalLength, uint32_t tileNum)
    {
        this->batch = batch;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->numGroups = numGroups;
        this->totalLength = totalLength;
        this->tileNum = tileNum;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        groupNormWeightGm.SetGlobalBuffer((__gm__ float *)groupNormWeight, outFeatures);
        groupNormBiasGm.SetGlobalBuffer((__gm__ float *)groupNormBias, outFeatures);
        outputGm.SetGlobalBuffer((__gm__ float *)output + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * inFeatures * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGroupNormWeight, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGroupNormBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueTemp, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> groupNormWeightLocal = inQueueGroupNormWeight.AllocTensor<float>();
        AscendC::LocalTensor<float> groupNormBiasLocal = inQueueGroupNormBias.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], this->tileLength * this->inFeatures);
        AscendC::DataCopy(biasLocal, biasGm[0], this->tileLength);
        AscendC::DataCopy(groupNormWeightLocal, groupNormWeightGm[0], this->tileLength);
        AscendC::DataCopy(groupNormBiasLocal, groupNormBiasGm[0], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
        inQueueGroupNormWeight.EnQue(groupNormWeightLocal);
        inQueueGroupNormBias.EnQue(groupNormBiasLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> groupNormWeightLocal = inQueueGroupNormWeight.DeQue<float>();
        AscendC::LocalTensor<float> groupNormBiasLocal = inQueueGroupNormBias.DeQue<float>();
        AscendC::LocalTensor<float> tempLocal = outQueueTemp.AllocTensor<float>();
        AscendC::LocalTensor<float> outputLocal = outQueueOutput.AllocTensor<float>();

        // Matrix multiply
        AscendC::MatMul(tempLocal, xLocal, weightLocal, this->tileLength, this->inFeatures, this->outFeatures, false, true);
        
        // Add bias
        AscendC::Add(outputLocal, tempLocal, biasLocal, this->tileLength);
        
        // Swish activation
        AscendC::Sigmoid(outputLocal, outputLocal, this->tileLength);
        AscendC::Mul(outputLocal, outputLocal, tempLocal, this->tileLength);
        
        // GroupNorm
        AscendC::GroupNorm(outputLocal, outputLocal, groupNormWeightLocal, groupNormBiasLocal, this->tileLength, this->numGroups);
        
        outQueueOutput.EnQue<float>(outputLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
        inQueueGroupNormWeight.FreeTensor(groupNormWeightLocal);
        inQueueGroupNormBias.FreeTensor(groupNormBiasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outputLocal = outQueueOutput.DeQue<float>();
        AscendC::DataCopy(outputGm[progress * this->tileLength], outputLocal, this->tileLength);
        outQueueOutput.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias, inQueueGroupNormWeight, inQueueGroupNormBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueTemp, outQueueOutput;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> groupNormWeightGm;
    AscendC::GlobalTensor<float> groupNormBiasGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t batch;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t numGroups;
    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_swish_sum_group_norm_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR groupNormWeight, GM_ADDR groupNormBias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulSwishSumGroupNorm op;
    op.Init(x, weight, bias, groupNormWeight, groupNormBias, output, tiling_data.batch, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.numGroups, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
