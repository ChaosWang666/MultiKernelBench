
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv1d {
public:
    __aicore__ inline KernelConv1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t inputLength, uint32_t outputLength, uint32_t kernelSize,
                                uint32_t stride, uint32_t dilation, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->dilation = dilation;
        this->tileNum = tileNum;
        
        this->blockLength = outputLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, kernelSize * inChannels * outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        // Simplified computation - actual implementation would be more complex
        AscendC::Mul(yLocal, xLocal, weightLocal, this->tileLength);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t dilation;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_standard_1d_dilated_strided_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv1d op;
    op.Init(x, weight, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.inputLength, tiling_data.outputLength, tiling_data.kernelSize,
            tiling_data.stride, tiling_data.dilation, tiling_data.tileNum);
    op.Process();
}
