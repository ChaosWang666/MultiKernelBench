
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv1d {
public:
    __aicore__ inline KernelConv1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelSize, uint32_t inputLength, uint32_t outputLength,
                                uint32_t stride, uint32_t padding, uint32_t dilation, uint32_t groups)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->groups = groups;

        this->blockLength = inputLength * inChannels * batchSize / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inChannels * outChannels * kernelSize / groups);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outChannels * outputLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, inChannels * outChannels * kernelSize / groups * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->outputLength * this->batchSize * this->outChannels / BUFFER_NUM;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], inChannels * outChannels * kernelSize / groups * sizeof(DTYPE_WEIGHT));
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simplified convolution computation
        AscendC::Conv1d(yLocal, xLocal, weightLocal, this->inputLength, this->outputLength, this->kernelSize, this->stride, this->padding, this->dilation, this->groups);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t kernelSize;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t groups;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_standard_1d_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv1d op;
    op.Init(x, weight, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.kernelSize, tiling_data.inputLength, tiling_data.outputLength,
            tiling_data.stride, tiling_data.padding, tiling_data.dilation, tiling_data.groups);
    op.Process();
}
