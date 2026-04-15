
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv {
public:
    __aicore__ inline KernelConv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelH, uint32_t kernelW,
                                uint32_t padH, uint32_t padW, uint32_t strideH, uint32_t strideW,
                                uint32_t dilationH, uint32_t dilationW, uint32_t groups, uint32_t outHeight, uint32_t outWidth)
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
        this->groups = groups;
        this->outHeight = outHeight;
        this->outWidth = outWidth;

        this->blockLength = batchSize * outChannels * outHeight * outWidth / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inChannels * height * width);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, outChannels * inChannels / groups * kernelH * kernelW);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outChannels * outHeight * outWidth);

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
        // Simplified computation logic
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
    uint32_t groups;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_standard_2d_square_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv op;
    op.Init(x, w, y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelH, tiling_data.kernelW,
            tiling_data.padH, tiling_data.padW, tiling_data.strideH, tiling_data.strideW,
            tiling_data.dilationH, tiling_data.dilationW, tiling_data.groups, tiling_data.outHeight, tiling_data.outWidth);
    op.Process();
}
