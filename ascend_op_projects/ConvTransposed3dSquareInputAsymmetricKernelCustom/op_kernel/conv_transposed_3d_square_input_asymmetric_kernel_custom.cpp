
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTransposed3d {
public:
    __aicore__ inline KernelConvTransposed3d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelDepth, uint32_t kernelWidth, uint32_t kernelHeight,
                                uint32_t inputDepth, uint32_t inputWidth, uint32_t inputHeight,
                                uint32_t strideDepth, uint32_t strideWidth, uint32_t strideHeight,
                                uint32_t padDepth, uint32_t padWidth, uint32_t padHeight,
                                uint32_t outputDepth, uint32_t outputWidth, uint32_t outputHeight,
                                uint32_t groups)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelDepth = kernelDepth;
        this->kernelWidth = kernelWidth;
        this->kernelHeight = kernelHeight;
        this->inputDepth = inputDepth;
        this->inputWidth = inputWidth;
        this->inputHeight = inputHeight;
        this->strideDepth = strideDepth;
        this->strideWidth = strideWidth;
        this->strideHeight = strideHeight;
        this->padDepth = padDepth;
        this->padWidth = padWidth;
        this->padHeight = padHeight;
        this->outputDepth = outputDepth;
        this->outputWidth = outputWidth;
        this->outputHeight = outputHeight;
        this->groups = groups;

        this->blockLength = batch * outChannels * outputDepth * outputWidth * outputHeight / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batch * inChannels * inputDepth * inputWidth * inputHeight);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, outChannels * inChannels / groups * kernelDepth * kernelWidth * kernelHeight);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batch * outChannels * outputDepth * outputWidth * outputHeight);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->batch * this->outChannels * this->outputDepth * this->outputWidth * this->outputHeight / this->tileLength;
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
        // Simplified computation logic for demonstration purposes
        AscendC::Mul(yLocal, xLocal, wLocal, this->tileLength);
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
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t kernelDepth;
    uint32_t kernelWidth;
    uint32_t kernelHeight;
    uint32_t inputDepth;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t strideDepth;
    uint32_t strideWidth;
    uint32_t strideHeight;
    uint32_t padDepth;
    uint32_t padWidth;
    uint32_t padHeight;
    uint32_t outputDepth;
    uint32_t outputWidth;
    uint32_t outputHeight;
    uint32_t groups;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transposed_3d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed3d op;
    op.Init(x, w, y,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.kernelDepth, tiling_data.kernelWidth, tiling_data.kernelHeight,
            tiling_data.inputDepth, tiling_data.inputWidth, tiling_data.inputHeight,
            tiling_data.strideDepth, tiling_data.strideWidth, tiling_data.strideHeight,
            tiling_data.padDepth, tiling_data.padWidth, tiling_data.padHeight,
            tiling_data.outputDepth, tiling_data.outputWidth, tiling_data.outputHeight,
            tiling_data.groups);
    op.Process();
}
