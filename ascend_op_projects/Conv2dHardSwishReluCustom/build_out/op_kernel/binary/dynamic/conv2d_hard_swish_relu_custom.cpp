
#include "kernel_operator.h"

class KernelConv2dHardSwishRelu {
public:
    __aicore__ inline KernelConv2dHardSwishRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                 uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                 uint32_t height, uint32_t width, uint32_t kernelSize,
                                 uint32_t totalTasks, uint32_t tasksPerCore)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->outHeight = height - kernelSize + 1;
        this->outWidth = width - kernelSize + 1;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->startTask = blockIdx * tasksPerCore;
        uint32_t endCandidate = this->startTask + tasksPerCore;
        this->endTask = (endCandidate < totalTasks) ? endCandidate : totalTasks;
        if (this->startTask > totalTasks) this->startTask = totalTasks;

        uint32_t outBytes = this->outWidth * sizeof(float);
        this->outBufferSize = ((outBytes + 31) / 32) * 32;

        xGm.SetGlobalBuffer((__gm__ float*)x);
        wGm.SetGlobalBuffer((__gm__ float*)weight);
        bGm.SetGlobalBuffer((__gm__ float*)bias);
        yGm.SetGlobalBuffer((__gm__ float*)y);

        uint32_t inputSize = this->inChannels * this->kernelSize * this->width * sizeof(float);
        inputSize = ((inputSize + 31) / 32) * 32;
        uint32_t weightSize = this->outChannels * this->inChannels * this->kernelSize * this->kernelSize * sizeof(float);
        weightSize = ((weightSize + 31) / 32) * 32;
        uint32_t biasSize = ((this->outChannels * sizeof(float) + 31) / 32) * 32;

        pipe.InitBuffer(inQueueW, 1, weightSize);
        pipe.InitBuffer(inQueueB, 1, biasSize);
        pipe.InitBuffer(inQueueX, 1, inputSize);
        pipe.InitBuffer(outQueueY, 1, this->outBufferSize);
        pipe.InitBuffer(accumBuf, this->outBufferSize);
        pipe.InitBuffer(tmpBuf, this->outBufferSize);
    }

    __aicore__ inline void Process()
    {
        if (startTask >= endTask) return;

        AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = outChannels * inChannels * kernelSize * kernelSize * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(wLocal, wGm, copyParams, padParams);
        }
        inQueueW.EnQue(wLocal);
        wLocal = inQueueW.DeQue<float>();

        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = outChannels * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(bLocal, bGm, copyParams, padParams);
        }
        inQueueB.EnQue(bLocal);
        bLocal = inQueueB.DeQue<float>();

        for (uint32_t task = startTask; task < endTask; task++) {
            ProcessTask(task, wLocal, bLocal);
        }

        inQueueW.FreeTensor(wLocal);
        inQueueB.FreeTensor(bLocal);
    }

private:
    __aicore__ inline void ProcessTask(uint32_t task,
                                        AscendC::LocalTensor<float> wLocal,
                                        AscendC::LocalTensor<float> bLocal)
    {
        uint32_t batchIdx = task / outHeight;
        uint32_t ohIdx = task % outHeight;
        int32_t outElemCount = static_cast<int32_t>(outWidth);

        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        {
            uint32_t srcOffset = batchIdx * inChannels * height * width + ohIdx * width;
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(inChannels);
            copyParams.blockLen = kernelSize * width * sizeof(float);
            copyParams.srcStride = (height - kernelSize) * width * sizeof(float);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(xLocal, xGm[srcOffset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();

        AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();

        uint32_t outBatchStride = outChannels * outHeight * outWidth;
        uint32_t outChanStride = outHeight * outWidth;

        for (uint32_t ocIdx = 0; ocIdx < outChannels; ocIdx++) {
            float biasVal = bLocal.GetValue(ocIdx);
            AscendC::Duplicate<float>(accum, biasVal, outElemCount);

            uint32_t wBase = ocIdx * inChannels * kernelSize * kernelSize;
            for (uint32_t ic = 0; ic < inChannels; ic++) {
                uint32_t xChanOffset = ic * kernelSize * width;
                uint32_t wChanBase = wBase + ic * kernelSize * kernelSize;
                for (uint32_t kh = 0; kh < kernelSize; kh++) {
                    uint32_t xRowOffset = xChanOffset + kh * width;
                    uint32_t wRowBase = wChanBase + kh * kernelSize;
                    for (uint32_t kw = 0; kw < kernelSize; kw++) {
                        float wVal = wLocal.GetValue(wRowBase + kw);
                        AscendC::Muls<float>(tmp, xLocal[xRowOffset + kw], wVal, outElemCount);
                        AscendC::Add<float>(accum, accum, tmp, outElemCount);
                    }
                }
            }

            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            AscendC::Adds<float>(tmp, accum, 3.0f, outElemCount);
            AscendC::Relu<float>(tmp, tmp, outElemCount);
            AscendC::Mins<float>(tmp, tmp, 6.0f, outElemCount);
            AscendC::Muls<float>(tmp, tmp, 1.0f / 6.0f, outElemCount);
            AscendC::Mul<float>(yLocal, accum, tmp, outElemCount);
            AscendC::Relu<float>(yLocal, yLocal, outElemCount);
            outQueueY.EnQue(yLocal);

            yLocal = outQueueY.DeQue<float>();
            {
                uint32_t yOffset = batchIdx * outBatchStride + ocIdx * outChanStride + ohIdx * outWidth;
                AscendC::DataCopyExtParams copyOut;
                copyOut.blockCount = 1;
                copyOut.blockLen = outWidth * sizeof(float);
                copyOut.srcStride = 0;
                copyOut.dstStride = 0;
                copyOut.rsv = 0;
                AscendC::DataCopyPad(yGm[yOffset], yLocal, copyOut);
            }
            outQueueY.FreeTensor(yLocal);
        }

        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueW, inQueueB, inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf, tmpBuf;
    AscendC::GlobalTensor<float> xGm, wGm, bGm, yGm;

    uint32_t batchSize, inChannels, outChannels, height, width, kernelSize;
    uint32_t outHeight, outWidth, outBufferSize;
    uint32_t startTask, endTask;
};

extern "C" __global__ __aicore__ void conv2d_hard_swish_relu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dHardSwishRelu op;
    op.Init(x, weight, bias, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize,
            tiling_data.totalTasks, tiling_data.tasksPerCore);
    op.Process();
}
