
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv2dTanhScalingBiasAddMax {
public:
    __aicore__ inline KernelConv2dTanhScalingBiasAddMax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelH, uint32_t kernelW,
                                uint32_t padH, uint32_t padW, uint32_t strideH, uint32_t strideW,
                                uint32_t dilationH, uint32_t dilationW, float scalingFactor,
                                uint32_t poolKernelH, uint32_t poolKernelW)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelH = kernelH;
        this->kernelW = kernelW;
        this->padH = padH;
        this->padW = padW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->scalingFactor = scalingFactor;
        this->poolKernelH = poolKernelH;
        this->poolKernelW = poolKernelW;

        this->blockLength = batchSize * outChannels * height * width;
        this->tileLength = this->blockLength / AscendC::GetBlockNum() / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inChannels * outChannels * kernelH * kernelW);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outChannels);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        // Convolution operation
        AscendC::Conv2d(zLocal, xLocal, weightLocal, biasLocal, this->height, this->width, this->kernelH, this->kernelW,
                        this->padH, this->padW, this->strideH, this->strideW, this->dilationH, this->dilationW);

        // Apply tanh
        AscendC::Tanh(zLocal, zLocal, this->tileLength);

        // Scale
        AscendC::Mul(zLocal, zLocal, this->scalingFactor, this->tileLength);

        // Add bias
        AscendC::Add(zLocal, zLocal, biasLocal, this->tileLength);

        // Max pooling
        AscendC::MaxPool2d(zLocal, zLocal, this->height, this->width, this->poolKernelH, this->poolKernelW);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t padH;
    uint32_t padW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    float scalingFactor;
    uint32_t poolKernelH;
    uint32_t poolKernelW;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_tanh_scaling_bias_add_max_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dTanhScalingBiasAddMax op;
    op.Init(x, weight, bias, z,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelH, tiling_data.kernelW,
            tiling_data.padH, tiling_data.padW, tiling_data.strideH, tiling_data.strideW,
            tiling_data.dilationH, tiling_data.dilationW, tiling_data.scalingFactor,
            tiling_data.poolKernelH, tiling_data.poolKernelW);
    op.Process();
}
