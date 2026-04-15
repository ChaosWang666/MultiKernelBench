
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv2dReluHardSwish {
public:
    __aicore__ inline KernelConv2dReluHardSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                uint32_t batchSize, uint32_t inputChannels, uint32_t outputChannels,
                                uint32_t inputHeight, uint32_t inputWidth,
                                uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t padH, uint32_t padW,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t dilationH, uint32_t dilationW)
    {
        this->batchSize = batchSize;
        this->inputChannels = inputChannels;
        this->outputChannels = outputChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->padH = padH;
        this->padW = padW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->dilationH = dilationH;
        this->dilationW = dilationW;

        this->outputHeight = (inputHeight + 2 * padH - (dilationH * (kernelHeight - 1) + 1)) / strideH + 1;
        this->outputWidth = (inputWidth + 2 * padW - (dilationW * (kernelWidth - 1) + 1)) / strideW + 1;

        this->blockLength = batchSize * outputChannels * outputHeight * outputWidth;
        this->tileNum = 1024;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * inputChannels * inputHeight * inputWidth);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outputChannels * inputChannels * kernelHeight * kernelWidth);
        if (bias != 0) {
            biasGm.SetGlobalBuffer((__gm__ float *)bias, outputChannels);
        } else {
            biasGm.SetGlobalBuffer(nullptr, 0);
        }
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * outputChannels * outputHeight * outputWidth);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        if (biasGm.GetGlobalBuffer() != nullptr) {
            AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);
        } else {
            // Initialize biasLocal to zero
            AscendC::Fill(biasLocal, 0.0f);
        }

        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Perform convolution operation
        AscendC::Conv2d(yLocal, xLocal, weightLocal, biasLocal, this->inputHeight, this->inputWidth,
                        this->outputHeight, this->outputWidth, this->kernelHeight, this->kernelWidth,
                        this->padH, this->padW, this->strideH, this->strideW, this->dilationH, this->dilationW);

        // Apply ReLU
        AscendC::ReLU(yLocal, yLocal, this->tileLength);

        // Apply HardSwish: x * clamp((x + 3) / 6, 0, 1)
        AscendC::LocalTensor<float> tempLocal = outQueueY.AllocTensor<float>();
        AscendC::Add(tempLocal, yLocal, 3.0f, this->tileLength); // x + 3
        AscendC::Div(tempLocal, tempLocal, 6.0f, this->tileLength); // (x + 3) / 6
        AscendC::Clamp(tempLocal, tempLocal, 0.0f, 1.0f, this->tileLength); // clamp((x + 3) / 6, 0, 1)
        AscendC::Mul(yLocal, yLocal, tempLocal, this->tileLength); // x * clamp((x + 3) / 6, 0, 1)

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t batchSize;
    uint32_t inputChannels;
    uint32_t outputChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t padH;
    uint32_t padW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_relu_hard_swish_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dReluHardSwish op;
    op.Init(x, weight, bias, y,
            tiling_data.batchSize,
            tiling_data.inputChannels,
            tiling_data.outputChannels,
            tiling_data.inputHeight,
            tiling_data.inputWidth,
            tiling_data.kernelHeight,
            tiling_data.kernelWidth,
            tiling_data.padH,
            tiling_data.padW,
            tiling_data.strideH,
            tiling_data.strideW,
            tiling_data.dilationH,
            tiling_data.dilationW);
    op.Process();
}
