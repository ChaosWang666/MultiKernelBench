
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv3dScalingTanhMultiplySigmoid {
public:
    __aicore__ inline KernelConv3dScalingTanhMultiplySigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR y,
        uint32_t totalChannels, uint32_t perChannelSize, uint32_t channels,
        uint32_t channelsPerCore, uint32_t tailChannels, uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startCh = 0;
        uint32_t endCh = 0;
        if (blockIdx < tailChannels) {
            startCh = blockIdx * (channelsPerCore + 1);
            endCh = startCh + channelsPerCore + 1;
        } else {
            startCh = tailChannels * (channelsPerCore + 1) +
                      (blockIdx - tailChannels) * channelsPerCore;
            endCh = startCh + channelsPerCore;
        }
        this->startChannel = startCh;
        this->endChannel = endCh;
        this->perChannelSize = perChannelSize;
        this->channels = channels;
        this->tileLength = tileLength;

        uint64_t totalElements = (uint64_t)totalChannels * (uint64_t)perChannelSize;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalElements);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElements);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, channels);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, channels);

        uint32_t scaleBytes = channels * sizeof(float);
        uint32_t alignedScaleBytes = ((scaleBytes + 31) / 32) * 32;
        if (alignedScaleBytes < 32) {
            alignedScaleBytes = 32;
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(scaleBuf, alignedScaleBytes);
        pipe.InitBuffer(biasBuf, alignedScaleBytes);
    }

    __aicore__ inline void Process()
    {
        if (startChannel >= endChannel) {
            return;
        }

        AscendC::LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();

        AscendC::DataCopyExtParams cpParams;
        cpParams.blockCount = 1;
        cpParams.blockLen = channels * sizeof(float);
        cpParams.srcStride = 0;
        cpParams.dstStride = 0;
        cpParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        AscendC::DataCopyPad(scaleLocal, scaleGm, cpParams, padParams);
        AscendC::DataCopyPad(biasLocal, biasGm, cpParams, padParams);

        auto eventMte2ToS = pipe.FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventMte2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventMte2ToS);

        for (uint32_t ch = startChannel; ch < endChannel; ch++) {
            uint32_t channelIdx = ch % channels;
            float scaleVal = scaleLocal.GetValue(channelIdx);
            float biasVal = biasLocal.GetValue(channelIdx);

            uint64_t chOffset = (uint64_t)ch * (uint64_t)perChannelSize;
            uint32_t processed = 0;
            while (processed < perChannelSize) {
                uint32_t remaining = perChannelSize - processed;
                uint32_t curLen = (remaining < tileLength) ? remaining : tileLength;
                CopyIn(chOffset + processed, curLen);
                Compute(curLen, scaleVal, biasVal);
                CopyOut(chOffset + processed, curLen);
                processed += curLen;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = length * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length, float scaleVal, float biasVal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        AscendC::Muls(yLocal, xLocal, scaleVal, (int32_t)length);
        AscendC::Tanh(yLocal, yLocal, (int32_t)length);
        AscendC::Muls(yLocal, yLocal, biasVal, (int32_t)length);
        AscendC::Sigmoid(yLocal, yLocal, (int32_t)length);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = length * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;

    uint32_t startChannel;
    uint32_t endChannel;
    uint32_t perChannelSize;
    uint32_t channels;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv3d_scaling_tanh_multiply_sigmoid_custom(
    GM_ADDR x, GM_ADDR scale, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dScalingTanhMultiplySigmoid op;
    op.Init(x, scale, bias, y,
        tiling_data.totalChannels, tiling_data.perChannelSize, tiling_data.channels,
        tiling_data.channelsPerCore, tiling_data.tailChannels, tiling_data.tileLength);
    op.Process();
}
