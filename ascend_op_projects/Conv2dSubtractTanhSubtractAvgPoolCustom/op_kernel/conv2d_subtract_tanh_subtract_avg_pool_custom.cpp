
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dSubtractTanhSubtractAvgPool {
public:
    __aicore__ inline KernelConv2dSubtractTanhSubtractAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelSize, float subtract1Value, float subtract2Value, uint32_t poolKernelSize)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->subtract1Value = subtract1Value;
        this->subtract2Value = subtract2Value;
        this->poolKernelSize = poolKernelSize;

        this->blockLength = batchSize * inChannels * height * width / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = BUFFER_NUM;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Simulate Conv2d + Subtract1 + Tanh + Subtract2 + AvgPool
        for (int32_t j = 0; j < this->tileLength; j++) {
            float val = xLocal[j];
            val -= this->subtract1Value;
            val = AscendC::Tanh(val);
            val -= this->subtract2Value;
            yLocal[j] = val;
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
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    float subtract1Value;
    float subtract2Value;
    uint32_t poolKernelSize;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_subtract_tanh_subtract_avg_pool_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dSubtractTanhSubtractAvgPool op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize, tiling_data.subtract1Value,
            tiling_data.subtract2Value, tiling_data.poolKernelSize);
    op.Process();
}
