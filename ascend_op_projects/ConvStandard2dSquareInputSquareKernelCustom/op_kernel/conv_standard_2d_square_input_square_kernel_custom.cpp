
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv {
public:
    __aicore__ inline KernelConv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                uint32_t batchSize, uint32_t inputHeight, uint32_t inputWidth,
                                uint32_t inputChannel, uint32_t outputChannel,
                                uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t padH, uint32_t padW,
                                uint32_t dilationH, uint32_t dilationW,
                                uint32_t outputHeight, uint32_t outputWidth,
                                uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inputHeight = inputHeight;
        this->inputWidth = inputWidth;
        this->inputChannel = inputChannel;
        this->outputChannel = outputChannel;
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
        this->tileNum = tileNum;

        this->blockLength = batchSize * outputHeight * outputWidth * outputChannel;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inputChannel * inputHeight * inputWidth);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, outputChannel * inputChannel * kernelHeight * kernelWidth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outputChannel * outputHeight * outputWidth);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(wLocal, wGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::Conv2d(yLocal, xLocal, wLocal, this->tileLength, this->inputHeight, this->inputWidth,
                        this->inputChannel, this->outputChannel, this->kernelHeight, this->kernelWidth,
                        this->strideH, this->strideW, this->padH, this->padW, this->dilationH, this->dilationW);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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
    uint32_t inputHeight;
    uint32_t inputWidth;
    uint32_t inputChannel;
    uint32_t outputChannel;
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
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_standard_2d_square_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv op;
    op.Init(x, w, y,
            tiling_data.batchSize,
            tiling_data.inputHeight,
            tiling_data.inputWidth,
            tiling_data.inputChannel,
            tiling_data.outputChannel,
            tiling_data.kernelHeight,
            tiling_data.kernelWidth,
            tiling_data.strideH,
            tiling_data.strideW,
            tiling_data.padH,
            tiling_data.padW,
            tiling_data.dilationH,
            tiling_data.dilationW,
            tiling_data.outputHeight,
            tiling_data.outputWidth,
            tiling_data.tileNum);
    op.Process();
}
