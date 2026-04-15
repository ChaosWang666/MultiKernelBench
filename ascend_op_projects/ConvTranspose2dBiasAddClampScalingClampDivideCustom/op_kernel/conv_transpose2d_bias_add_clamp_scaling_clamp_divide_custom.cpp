
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTranspose2dBiasAddClampScalingClampDivide {
public:
    __aicore__ inline KernelConvTranspose2dBiasAddClampScalingClampDivide() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width,
                                uint32_t kernelH, uint32_t kernelW,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t padH, uint32_t padW,
                                uint32_t outputPadH, uint32_t outputPadW,
                                float scalingFactor)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelH = kernelH;
        this->kernelW = kernelW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->padH = padH;
        this->padW = padW;
        this->outputPadH = outputPadH;
        this->outputPadW = outputPadW;
        this->scalingFactor = scalingFactor;

        this->blockLength = batchSize * outChannels * height * width / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inChannels * outChannels * kernelH * kernelW);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], inChannels * outChannels * kernelH * kernelW);
        AscendC::DataCopy(biasLocal, biasGm[0], outChannels);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // Perform ConvTranspose2d operation
        AscendC::ConvTranspose2d(yLocal, xLocal, weightLocal, biasLocal, 
                                 batchSize, inChannels, outChannels, 
                                 height, width, kernelH, kernelW, 
                                 strideH, strideW, padH, padW, 
                                 outputPadH, outputPadW);

        // Add bias
        AscendC::Add(yLocal, yLocal, biasLocal, this->tileLength);

        // Clamp [0.0, 1.0]
        AscendC::Clip(yLocal, yLocal, 0.0f, 1.0f, this->tileLength);

        // Scale
        AscendC::Mul(yLocal, yLocal, this->scalingFactor, this->tileLength);

        // Clamp again [0.0, 1.0]
        AscendC::Clip(yLocal, yLocal, 0.0f, 1.0f, this->tileLength);

        // Divide
        AscendC::Div(yLocal, yLocal, this->scalingFactor, this->tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t padH;
    uint32_t padW;
    uint32_t outputPadH;
    uint32_t outputPadW;
    float scalingFactor;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transpose2d_bias_add_clamp_scaling_clamp_divide_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dBiasAddClampScalingClampDivide op;
    op.Init(x, weight, bias, y,
            tiling_data.batchSize,
            tiling_data.inChannels,
            tiling_data.outChannels,
            tiling_data.height,
            tiling_data.width,
            tiling_data.kernelH,
            tiling_data.kernelW,
            tiling_data.strideH,
            tiling_data.strideW,
            tiling_data.padH,
            tiling_data.padW,
            tiling_data.outputPadH,
            tiling_data.outputPadW,
            tiling_data.scalingFactor);
    op.Process();
}
