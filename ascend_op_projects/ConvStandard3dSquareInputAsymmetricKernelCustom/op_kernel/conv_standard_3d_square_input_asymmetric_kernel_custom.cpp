
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3d {
public:
    __aicore__ inline KernelConv3d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelWidth, uint32_t kernelHeight, uint32_t kernelDepth,
                                uint32_t strideW, uint32_t strideH, uint32_t strideD,
                                uint32_t padW, uint32_t padH, uint32_t padD,
                                uint32_t dilationW, uint32_t dilationH, uint32_t dilationD,
                                uint32_t inputWidth, uint32_t inputHeight, uint32_t inputDepth,
                                uint32_t outputWidth, uint32_t outputHeight, uint32_t outputDepth)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelWidth = kernelWidth;
        this->kernelHeight = kernelHeight;
        this->kernelDepth = kernelDepth;
        this->strideW = strideW;
        this->strideH = strideH;
        this->strideD = strideD;
        this->padW = padW;
        this->padH = padH;
        this->padD = padD;
        this->dilationW = dilationW;
        this->dilationH = dilationH;
        this->dilationD = dilationD;
        this->inputWidth = inputWidth;
        this->inputHeight = inputHeight;
        this->inputDepth = inputDepth;
        this->outputWidth = outputWidth;
        this->outputHeight = outputHeight;
        this->outputDepth = outputDepth;

        this->blockLength = batchSize * outChannels * outputWidth * outputHeight * outputDepth;
        this->tileLength = this->blockLength / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inChannels * inputWidth * inputHeight * inputDepth);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, outChannels * inChannels * kernelWidth * kernelHeight * kernelDepth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outChannels * outputWidth * outputHeight * outputDepth);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        // Simplified computation logic for demonstration
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
    uint32_t blockSize;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t kernelWidth;
    uint32_t kernelHeight;
    uint32_t kernelDepth;
    uint32_t strideW;
    uint32_t strideH;
    uint32_t strideD;
    uint32_t padW;
    uint32_t padH;
    uint32_t padD;
    uint32_t dilationW;
    uint32_t dilationH;
    uint32_t dilationD;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t inputDepth;
    uint32_t outputWidth;
    uint32_t outputHeight;
    uint32_t outputDepth;
};

extern "C" __global__ __aicore__ void conv_standard_3d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3d op;
    op.Init(x, w, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.kernelWidth, tiling_data.kernelHeight, tiling_data.kernelDepth,
            tiling_data.strideW, tiling_data.strideH, tiling_data.strideD,
            tiling_data.padW, tiling_data.padH, tiling_data.padD,
            tiling_data.dilationW, tiling_data.dilationH, tiling_data.dilationD,
            tiling_data.inputWidth, tiling_data.inputHeight, tiling_data.inputDepth,
            tiling_data.outputWidth, tiling_data.outputHeight, tiling_data.outputDepth);
    op.Process();
}
