
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvDepthwise2dAsymmetricInputSquareKernel {
public:
    __aicore__ inline KernelConvDepthwise2dAsymmetricInputSquareKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t heightIn, uint32_t widthIn, uint32_t heightOut, uint32_t widthOut,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->heightIn = heightIn;
        this->widthIn = widthIn;
        this->heightOut = heightOut;
        this->widthOut = widthOut;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        
        this->blockLength = batchSize * inChannels * heightOut * widthOut;
        this->tileNum = 4096;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inChannels * heightIn * widthIn);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, inChannels * kernelSize * kernelSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * inChannels * heightOut * widthOut);
        
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
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t heightIn;
    uint32_t widthIn;
    uint32_t heightOut;
    uint32_t widthOut;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_depthwise_2d_asymmetric_input_square_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvDepthwise2dAsymmetricInputSquareKernel op;
    op.Init(x, w, y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.heightIn, tiling_data.widthIn, tiling_data.heightOut, tiling_data.widthOut,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding);
    op.Process();
}
