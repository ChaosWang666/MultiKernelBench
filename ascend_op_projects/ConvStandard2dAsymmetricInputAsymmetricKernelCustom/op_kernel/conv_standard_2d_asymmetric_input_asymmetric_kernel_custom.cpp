
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv2d {
public:
    __aicore__ inline KernelConv2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batchSize, uint32_t inputChannels, uint32_t outputChannels,
                                uint32_t inputHeight, uint32_t inputWidth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideH, uint32_t strideW, uint32_t padH, uint32_t padW, uint32_t dilationH, uint32_t dilationW,
                                uint32_t outputHeight, uint32_t outputWidth, uint32_t groups)
    {
        this->batchSize = batchSize;
        this->inputChannels = inputChannels;
        this->outputChannels = outputChannels;
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
        this->groups = groups;

        this->blockLength = batchSize * outputChannels * outputHeight * outputWidth;
        this->tileLength = this->blockLength / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, inputChannels * outputChannels * kernelHeight * kernelWidth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileLength;
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
        AscendC::DataCopy(xLocal, xGm[progress], 1);
        AscendC::DataCopy(wLocal, wGm[progress], 1);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::Mul(yLocal, xLocal, wLocal, 1);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress], yLocal, 1);
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
    uint32_t inputChannels;
    uint32_t outputChannels;
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
    uint32_t groups;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_standard_2d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2d op;
    op.Init(x, w, y, tiling_data.batchSize, tiling_data.inputChannels, tiling_data.outputChannels,
            tiling_data.inputHeight, tiling_data.inputWidth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.strideH, tiling_data.strideW, tiling_data.padH, tiling_data.padW, tiling_data.dilationH, tiling_data.dilationW,
            tiling_data.outputHeight, tiling_data.outputWidth, tiling_data.groups);
    op.Process();
}
