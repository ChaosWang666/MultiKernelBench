
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvDepthwise2dAsymmetricInputAsymmetricKernel {
public:
    __aicore__ inline KernelConvDepthwise2dAsymmetricInputAsymmetricKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t inputHeight, uint32_t inputWidth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideH, uint32_t strideW, uint32_t padH, uint32_t padW, uint32_t dilationH, uint32_t dilationW,
                                uint32_t outputHeight, uint32_t outputWidth, uint32_t tileNumH, uint32_t tileNumW)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->strideH = strideH;
        this->strideW = strideW;
        this->padH = padH;
        this->padW = padW;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->outputHeight = outputHeight;
        this->outputWidth = outputWidth;
        this->tileNumH = tileNumH;
        this->tileNumW = tileNumW;

        this->tileHeight = inputHeight / tileNumH;
        this->tileWidth = inputWidth / tileNumW;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inChannels * inputHeight * inputWidth);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, outChannels * inChannels * kernelHeight * kernelWidth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outChannels * outputHeight * outputWidth);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileHeight * tileWidth * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, kernelHeight * kernelWidth * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileHeight * tileWidth * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t channel = 0; channel < inChannels; ++channel) {
                for (uint32_t h = 0; h < outputHeight; h += tileNumH) {
                    for (uint32_t w = 0; w < outputWidth; w += tileNumW) {
                        CopyIn(batch, channel, h, w);
                        Compute(batch, channel, h, w);
                        CopyOut(batch, channel, h, w);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch, uint32_t channel, uint32_t h, uint32_t w)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        uint32_t xOffset = batch * inChannels * inputHeight * inputWidth + channel * inputHeight * inputWidth + h * inputWidth + w;
        uint32_t wOffset = channel * kernelHeight * kernelWidth;
        AscendC::DataCopy(xLocal, xGm[xOffset], tileHeight * tileWidth);
        AscendC::DataCopy(wLocal, wGm[wOffset], kernelHeight * kernelWidth);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
    }

    __aicore__ inline void Compute(uint32_t batch, uint32_t channel, uint32_t h, uint32_t w)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simplified computation logic for demonstration
        AscendC::Mul(yLocal, xLocal, wLocal, tileHeight * tileWidth);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batch, uint32_t channel, uint32_t h, uint32_t w)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t yOffset = batch * outChannels * outputHeight * outputWidth + channel * outputHeight * outputWidth + h * outputWidth + w;
        AscendC::DataCopy(yGm[yOffset], yLocal, tileHeight * tileWidth);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueW;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t padH;
    uint32_t padW;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t outputHeight;
    uint32_t outputWidth;
    uint32_t tileNumH;
    uint32_t tileNumW;
    uint32_t tileHeight;
    uint32_t tileWidth;
};

extern "C" __global__ __aicore__ void conv_depthwise_2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwise2dAsymmetricInputAsymmetricKernel op;
    op.Init(x, w, y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.inputHeight, tiling_data.inputWidth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.strideH, tiling_data.strideW, tiling_data.padH, tiling_data.padW, tiling_data.dilationH, tiling_data.dilationW,
            tiling_data.outputHeight, tiling_data.outputWidth, tiling_data.tileNumH, tiling_data.tileNumW);
    op.Process();
}
