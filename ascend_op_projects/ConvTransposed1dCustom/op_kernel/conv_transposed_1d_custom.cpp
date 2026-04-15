
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTransposed1d {
public:
    __aicore__ inline KernelConvTransposed1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding,
                                uint32_t outputPadding, uint32_t groups, uint32_t inputLength, uint32_t outputLength)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->outputPadding = outputPadding;
        this->groups = groups;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        
        this->blockLength = outputLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inChannels * inputLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, outChannels * inChannels / groups * kernelSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outChannels * outputLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            for (uint32_t oc = 0; oc < outChannels; ++oc) {
                for (uint32_t ic = 0; ic < inChannels; ++ic) {
                    for (uint32_t k = 0; k < kernelSize; ++k) {
                        CopyIn(batch, oc, ic, k);
                        Compute(batch, oc, ic, k);
                        CopyOut(batch, oc, ic, k);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch, uint32_t oc, uint32_t ic, uint32_t k)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::DataCopy(xLocal, xGm[batch * inChannels * inputLength + ic * inputLength], inputLength);
        AscendC::DataCopy(weightLocal, weightGm[oc * inChannels / groups * kernelSize + ic * kernelSize + k], kernelSize);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }
    
    __aicore__ inline void Compute(uint32_t batch, uint32_t oc, uint32_t ic, uint32_t k)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simplified computation for demonstration
        AscendC::Mul(yLocal, xLocal, weightLocal, inputLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t batch, uint32_t oc, uint32_t ic, uint32_t k)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[batch * outChannels * outputLength + oc * outputLength], yLocal, outputLength);
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
    uint32_t stride;
    uint32_t padding;
    uint32_t outputPadding;
    uint32_t groups;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transposed_1d_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed1d op;
    op.Init(x, weight, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding,
            tiling_data.outputPadding, tiling_data.groups, tiling_data.inputLength, tiling_data.outputLength);
    op.Process();
}
