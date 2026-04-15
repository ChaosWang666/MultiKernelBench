
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTransposed1d {
public:
    __aicore__ inline KernelConvTransposed1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y, uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding, uint32_t dilation, uint32_t inputLength, uint32_t outputLength)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->dilation = dilation;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->blockLength = outputLength / AscendC::GetBlockNum();
        this->blockStart = this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batch * inChannels * inputLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, outChannels * inChannels * kernelSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batch * outChannels * outputLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->blockLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t b = 0; b < batch; ++b) {
            for (uint32_t oc = 0; oc < outChannels; ++oc) {
                for (uint32_t ic = 0; ic < inChannels; ++ic) {
                    ProcessOneChannel(b, oc, ic);
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessOneChannel(uint32_t batchId, uint32_t outCh, uint32_t inCh)
    {
        uint32_t weightOffset = outCh * inChannels * kernelSize + inCh * kernelSize;
        uint32_t inputOffset = batchId * inChannels * inputLength + inCh * inputLength;
        uint32_t outputOffset = batchId * outChannels * outputLength + outCh * outputLength;

        for (uint32_t i = 0; i < inputLength; ++i) {
            uint32_t outputStart = i * stride - padding;
            for (uint32_t k = 0; k < kernelSize; ++k) {
                uint32_t outputPos = outputStart + k * dilation;
                if (outputPos >= 0 && outputPos < outputLength) {
                    float val = xGm[inputOffset + i];
                    float w = weightGm[weightOffset + k];
                    yGm[outputOffset + outputPos] += val * w;
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t dilation;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t blockLength;
    uint32_t blockStart;
};

extern "C" __global__ __aicore__ void conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed1d op;
    op.Init(x, weight, y, tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding, tiling_data.dilation,
            tiling_data.inputLength, tiling_data.outputLength);
    op.Process();
}
