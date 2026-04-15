
#include "kernel_operator.h"

class KernelAvgPool2d {
public:
    __aicore__ inline KernelAvgPool2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t batchSize, uint32_t channels,
                                uint32_t inputHeight, uint32_t inputWidth,
                                uint32_t outputHeight, uint32_t outputWidth,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;

        uint32_t totalPlanes = batchSize * channels;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->planesPerBlock = (totalPlanes + numBlocks - 1) / numBlocks;
        this->startPlane = blockIdx * this->planesPerBlock;
        if (this->startPlane > totalPlanes) this->startPlane = totalPlanes;
        this->endPlane = this->startPlane + this->planesPerBlock;
        if (this->endPlane > totalPlanes) this->endPlane = totalPlanes;

        uint32_t inPlaneSize = inputHeight * inputWidth;
        uint32_t outPlaneSize = outputHeight * outputWidth;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inPlaneSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channels * outPlaneSize);

        // Allocate buffer for one output row (outputWidth elements), aligned to 32 bytes
        uint32_t alignedOutWidth = ((outputWidth + 7) / 8) * 8;
        this->alignedOutWidth = alignedOutWidth;

        // Allocate buffer for one input row (inputWidth elements), aligned
        uint32_t alignedInWidth = ((inputWidth + 7) / 8) * 8;
        this->alignedInWidth = alignedInWidth;

        pipe.InitBuffer(outQueue, 1, alignedOutWidth * sizeof(float));
        pipe.InitBuffer(inQueue, 1, alignedInWidth * sizeof(float));
        pipe.InitBuffer(accumQueue, 1, alignedOutWidth * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t inPlaneSize = inputHeight * inputWidth;
        uint32_t outPlaneSize = outputHeight * outputWidth;

        for (uint32_t p = startPlane; p < endPlane; p++) {
            float* inBase = (float*)(xGm.GetPhyAddr()) + p * inPlaneSize;
            float* outBase = (float*)(yGm.GetPhyAddr()) + p * outPlaneSize;

            for (uint32_t oh = 0; oh < outputHeight; oh++) {
                // Compute the accumulation for this output row
                AscendC::LocalTensor<float> accumLocal = accumQueue.AllocTensor<float>();
                // Zero out accumulator
                AscendC::Duplicate(accumLocal, 0.0f, alignedOutWidth);

                int32_t hStart = (int32_t)(oh * stride) - (int32_t)padding;
                int32_t hEnd = hStart + (int32_t)kernelSize;
                if (hStart < 0) hStart = 0;
                if (hEnd > (int32_t)inputHeight) hEnd = (int32_t)inputHeight;

                for (int32_t ih = hStart; ih < hEnd; ih++) {
                    // For each output column, we need to sum over kernel width
                    // Process each output pixel
                    for (uint32_t ow = 0; ow < outputWidth; ow++) {
                        int32_t wStart = (int32_t)(ow * stride) - (int32_t)padding;
                        int32_t wEnd = wStart + (int32_t)kernelSize;
                        if (wStart < 0) wStart = 0;
                        if (wEnd > (int32_t)inputWidth) wEnd = (int32_t)inputWidth;

                        float sum = 0.0f;
                        uint32_t inRowOffset = p * inPlaneSize + ih * inputWidth;
                        for (int32_t iw = wStart; iw < wEnd; iw++) {
                            sum += (float)xGm.GetValue(inRowOffset + iw);
                        }
                        float prev = accumLocal.GetValue(ow);
                        accumLocal.SetValue(ow, prev + sum);
                    }
                }

                // Divide by kernel_size * kernel_size
                float invKernelArea = 1.0f / (float)(kernelSize * kernelSize);
                AscendC::Muls(accumLocal, accumLocal, invKernelArea, alignedOutWidth);

                // Copy out
                uint32_t outRowOffset = p * outPlaneSize + oh * outputWidth;
                // We need to write outputWidth elements
                // Use SetValue for safety with non-aligned sizes
                for (uint32_t ow = 0; ow < outputWidth; ow++) {
                    yGm.SetValue(outRowOffset + ow, accumLocal.GetValue(ow));
                }

                accumQueue.FreeTensor(accumLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> accumQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t batchSize, channels;
    uint32_t inputHeight, inputWidth;
    uint32_t outputHeight, outputWidth;
    uint32_t kernelSize, stride, padding;
    uint32_t planesPerBlock, startPlane, endPlane;
    uint32_t alignedOutWidth, alignedInWidth;
};

extern "C" __global__ __aicore__ void average_pooling_2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAvgPool2d op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.inputHeight, tiling_data.inputWidth,
            tiling_data.outputHeight, tiling_data.outputWidth,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding);
    op.Process();
}
