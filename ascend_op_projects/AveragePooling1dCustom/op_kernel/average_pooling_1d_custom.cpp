
#include "kernel_operator.h"

class KernelAvgPool1d {
public:
    __aicore__ inline KernelAvgPool1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t inputLength, uint32_t outputLength,
                                 uint32_t kernelSize, uint32_t stride, uint32_t padding)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;

        uint32_t totalChannels = batchSize * channels;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->channelsPerBlock = (totalChannels + numBlocks - 1) / numBlocks;
        this->startChannel = blockIdx * this->channelsPerBlock;
        if (this->startChannel > totalChannels) {
            this->startChannel = totalChannels;
        }
        this->endChannel = this->startChannel + this->channelsPerBlock;
        if (this->endChannel > totalChannels) {
            this->endChannel = totalChannels;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, (uint64_t)batchSize * channels * inputLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, (uint64_t)batchSize * channels * outputLength);

        // Allocate buffer for one input row + padding processed tile output
        // We'll process output in tiles
        uint32_t alignedInput = ((inputLength + 7) / 8) * 8;
        uint32_t alignedOutput = ((outputLength + 7) / 8) * 8;

        // We need buffers for input and output per channel
        // Use pipe for buffer management
        pipe.InitBuffer(inQueue, 1, alignedInput * sizeof(float));
        pipe.InitBuffer(outQueue, 1, alignedOutput * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t ch = this->startChannel; ch < this->endChannel; ch++) {
            CopyIn(ch);
            Compute(ch);
            CopyOut(ch);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t ch)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        uint32_t alignedInput = ((inputLength + 7) / 8) * 8;
        // Zero out the buffer first
        AscendC::Duplicate(xLocal, (float)0.0f, alignedInput);
        AscendC::DataCopy(xLocal, xGm[ch * (uint64_t)inputLength], inputLength);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t ch)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        uint32_t alignedOutput = ((outputLength + 7) / 8) * 8;
        AscendC::Duplicate(yLocal, (float)0.0f, alignedOutput);

        float invKernel = 1.0f / (float)kernelSize;

        // For each output position, compute average
        for (uint32_t o = 0; o < outputLength; o++) {
            int32_t startPos = (int32_t)(o * stride) - (int32_t)padding;
            float sum = 0.0f;
            for (uint32_t k = 0; k < kernelSize; k++) {
                int32_t inPos = startPos + (int32_t)k;
                if (inPos >= 0 && inPos < (int32_t)inputLength) {
                    sum += xLocal.GetValue(inPos);
                }
            }
            yLocal.SetValue(o, sum * invKernel);
        }

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t ch)
    {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(yGm[ch * (uint64_t)outputLength], yLocal, outputLength);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t channelsPerBlock;
    uint32_t startChannel;
    uint32_t endChannel;
};

extern "C" __global__ __aicore__ void average_pooling_1d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAvgPool1d op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.inputLength, tiling_data.outputLength,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding);
    op.Process();
}
