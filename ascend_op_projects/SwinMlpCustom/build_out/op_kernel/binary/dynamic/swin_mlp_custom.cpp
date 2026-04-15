
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelSwinMlp {
public:
    __aicore__ inline KernelSwinMlp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR fc1Weight, GM_ADDR fc1Bias, GM_ADDR fc2Weight, GM_ADDR fc2Bias, GM_ADDR y, 
                                uint32_t batchSize, uint32_t seqLen, uint32_t hiddenDim, uint32_t outDim)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->hiddenDim = hiddenDim;
        this->outDim = outDim;
        this->blockLength = seqLen * hiddenDim / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * seqLen * hiddenDim);
        fc1WeightGm.SetGlobalBuffer((__gm__ float *)fc1Weight, hiddenDim * outDim);
        fc1BiasGm.SetGlobalBuffer((__gm__ float *)fc1Bias, outDim);
        fc2WeightGm.SetGlobalBuffer((__gm__ float *)fc2Weight, outDim * hiddenDim);
        fc2BiasGm.SetGlobalBuffer((__gm__ float *)fc2Bias, hiddenDim);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * seqLen * hiddenDim);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueFc1Weight, BUFFER_NUM, this->tileLength * outDim * sizeof(float));
        pipe.InitBuffer(inQueueFc1Bias, BUFFER_NUM, outDim * sizeof(float));
        pipe.InitBuffer(inQueueFc2Weight, BUFFER_NUM, outDim * hiddenDim * sizeof(float));
        pipe.InitBuffer(inQueueFc2Bias, BUFFER_NUM, hiddenDim * sizeof(float));
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
        AscendC::LocalTensor<float> fc1WeightLocal = inQueueFc1Weight.AllocTensor<float>();
        AscendC::LocalTensor<float> fc1BiasLocal = inQueueFc1Bias.AllocTensor<float>();
        AscendC::LocalTensor<float> fc2WeightLocal = inQueueFc2Weight.AllocTensor<float>();
        AscendC::LocalTensor<float> fc2BiasLocal = inQueueFc2Bias.AllocTensor<float>();
        
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(fc1WeightLocal, fc1WeightGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(fc1BiasLocal, fc1BiasGm, outDim);
        AscendC::DataCopy(fc2WeightLocal, fc2WeightGm, outDim * hiddenDim);
        AscendC::DataCopy(fc2BiasLocal, fc2BiasGm, hiddenDim);
        
        inQueueX.EnQue(xLocal);
        inQueueFc1Weight.EnQue(fc1WeightLocal);
        inQueueFc1Bias.EnQue(fc1BiasLocal);
        inQueueFc2Weight.EnQue(fc2WeightLocal);
        inQueueFc2Bias.EnQue(fc2BiasLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> fc1WeightLocal = inQueueFc1Weight.DeQue<float>();
        AscendC::LocalTensor<float> fc1BiasLocal = inQueueFc1Bias.DeQue<float>();
        AscendC::LocalTensor<float> fc2WeightLocal = inQueueFc2Weight.DeQue<float>();
        AscendC::LocalTensor<float> fc2BiasLocal = inQueueFc2Bias.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // FC1 computation
        AscendC::MatMul(yLocal, xLocal, fc1WeightLocal, fc1BiasLocal, seqLen, hiddenDim, outDim, true, false);
        
        // GELU activation
        AscendC::Gelu(yLocal, yLocal, seqLen * outDim);
        
        // FC2 computation
        AscendC::MatMul(yLocal, yLocal, fc2WeightLocal, fc2BiasLocal, seqLen, outDim, hiddenDim, true, false);
        
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueFc1Weight.FreeTensor(fc1WeightLocal);
        inQueueFc1Bias.FreeTensor(fc1BiasLocal);
        inQueueFc2Weight.FreeTensor(fc2WeightLocal);
        inQueueFc2Bias.FreeTensor(fc2BiasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueFc1Weight, inQueueFc1Bias, inQueueFc2Weight, inQueueFc2Bias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> fc1WeightGm;
    AscendC::GlobalTensor<float> fc1BiasGm;
    AscendC::GlobalTensor<float> fc2WeightGm;
    AscendC::GlobalTensor<float> fc2BiasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t hiddenDim;
    uint32_t outDim;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void swin_mlp_custom(GM_ADDR x, GM_ADDR fc1Weight, GM_ADDR fc1Bias, GM_ADDR fc2Weight, GM_ADDR fc2Bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSwinMlp op;
    op.Init(x, fc1Weight, fc1Bias, fc2Weight, fc2Bias, y, tiling_data.batchSize, tiling_data.seqLen, tiling_data.hiddenDim, tiling_data.outDim);
    op.Process();
}
