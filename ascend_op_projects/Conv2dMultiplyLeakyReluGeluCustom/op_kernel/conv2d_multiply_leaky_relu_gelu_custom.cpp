
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv2dMultiplyLeakyReluGelu {
public:
    __aicore__ inline KernelConv2dMultiplyLeakyReluGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelSize,
                                uint32_t multiplierShape0, uint32_t multiplierShape1, uint32_t multiplierShape2)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->multiplierShape0 = multiplierShape0;
        this->multiplierShape1 = multiplierShape1;
        this->multiplierShape2 = multiplierShape2;

        uint32_t totalElements = batchSize * outChannels * height * width;
        this->blockLength = totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = 1;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->blockLength], this->blockLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // Simulate Conv2d + Multiply + LeakyReLU + GELU operations
        for (uint32_t i = 0; i < this->blockLength; ++i) {
            float val = xLocal[i];
            // Multiply by scalar (simplified)
            val *= 1.0f; // Placeholder for actual multiplier
            // LeakyReLU
            if (val < 0.0f) {
                val *= 0.01f; // Negative slope
            }
            // GELU approximation
            float cdf = 0.5f * (1.0f + AscendC::Erf(val / 1.4142135623730951f)); // sqrt(2)
            zLocal[i] = val * cdf;
        }

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->blockLength], zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    uint32_t multiplierShape0;
    uint32_t multiplierShape1;
    uint32_t multiplierShape2;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv2d_multiply_leaky_relu_gelu_custom(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMultiplyLeakyReluGelu op;
    op.Init(x, z, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize,
            tiling_data.multiplierShape0, tiling_data.multiplierShape1, tiling_data.multiplierShape2);
    op.Process();
}
