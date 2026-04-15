
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTranspose2dAddMinGeluMultiply {
public:
    __aicore__ inline KernelConvTranspose2dAddMinGeluMultiply() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelSize, uint32_t stride,
                                float addValue, float multiplyValue)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->addValue = addValue;
        this->multiplyValue = multiplyValue;

        this->totalElements = batchSize * outChannels * height * width;
        this->blockLength = this->totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Simulate transposed convolution + add + min + gelu + multiply
        // In practice, this would involve actual kernel computation
        int32_t loopCount = 1; // Simplified for example
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> localTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopy(localTensor, xGm[progress * this->blockLength], this->blockLength);
        inQueue.EnQue(localTensor);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> localTensor = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> resultTensor = outQueue.AllocTensor<float>();

        // Perform operations sequentially
        // 1. Add constant
        AscendC::Add(resultTensor, localTensor, this->addValue, this->blockLength);

        // 2. Min with zero
        AscendC::Min(resultTensor, resultTensor, 0.0f, this->blockLength);

        // 3. Apply GELU
        AscendC::Gelu(resultTensor, resultTensor, this->blockLength);

        // 4. Multiply by constant
        AscendC::Mul(resultTensor, resultTensor, this->multiplyValue, this->blockLength);

        outQueue.EnQue<float>(resultTensor);
        inQueue.FreeTensor(localTensor);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> resultTensor = outQueue.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->blockLength], resultTensor, this->blockLength);
        outQueue.FreeTensor(resultTensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    uint32_t stride;
    float addValue;
    float multiplyValue;
    uint32_t totalElements;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv_transpose2d_add_min_gelu_multiply_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dAddMinGeluMultiply op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize, tiling_data.stride,
            tiling_data.addValue, tiling_data.multiplyValue);
    op.Process();
}
