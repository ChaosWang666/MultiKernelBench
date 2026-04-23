
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dMultiplyLeakyReluGelu {
public:
    __aicore__ inline KernelConv2dMultiplyLeakyReluGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR multiplier, GM_ADDR z,
                                uint32_t totalChannels,
                                uint32_t channelSize,
                                uint32_t channels,
                                uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t channelsPerBlock = totalChannels / blockNum;
        uint32_t remainder = totalChannels % blockNum;

        if (blockIdx < remainder) {
            this->startChannel = blockIdx * (channelsPerBlock + 1);
            this->blockChannels = channelsPerBlock + 1;
        } else {
            this->startChannel = blockIdx * channelsPerBlock + remainder;
            this->blockChannels = channelsPerBlock;
        }

        this->channelSize = channelSize;
        this->channels = channels;
        this->tileLength = tileLength;

        uint64_t gmStart = (uint64_t)this->startChannel * (uint64_t)channelSize;
        uint64_t gmLen = (uint64_t)this->blockChannels * (uint64_t)channelSize;

        xGm.SetGlobalBuffer((__gm__ float *)x + gmStart, gmLen);
        zGm.SetGlobalBuffer((__gm__ float *)z + gmStart, gmLen);
        multiplierGm.SetGlobalBuffer((__gm__ float *)multiplier, channels);

        uint32_t multBufSize = ((channels * sizeof(float) + 31) / 32) * 32;
        pipe.InitBuffer(multQueue, 1, multBufSize);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (this->blockChannels == 0) return;

        AscendC::LocalTensor<float> multTmp = multQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams multCopy;
        multCopy.blockCount = 1;
        multCopy.blockLen = this->channels * sizeof(float);
        multCopy.srcStride = 0;
        multCopy.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> multPad;
        multPad.isPad = false;
        multPad.leftPadding = 0;
        multPad.rightPadding = 0;
        multPad.paddingValue = 0;
        AscendC::DataCopyPad(multTmp, multiplierGm, multCopy, multPad);
        multQueue.EnQue(multTmp);
        AscendC::LocalTensor<float> multLocal = multQueue.DeQue<float>();

        for (uint32_t c = 0; c < this->blockChannels; c++) {
            uint32_t globalChannelIdx = (this->startChannel + c) % this->channels;
            float mult = multLocal.GetValue(globalChannelIdx);

            uint32_t offset = 0;
            while (offset < this->channelSize) {
                uint32_t remain = this->channelSize - offset;
                uint32_t curLen = (remain < this->tileLength) ? remain : this->tileLength;
                CopyIn(c, offset, curLen);
                Compute(curLen, mult);
                CopyOut(c, offset, curLen);
                offset += curLen;
            }
        }

        multQueue.FreeTensor(multLocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t c, uint32_t offset, uint32_t len) {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        uint64_t gmOffset = (uint64_t)c * (uint64_t)this->channelSize + (uint64_t)offset;
        AscendC::DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len, float mult) {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        int32_t n = (int32_t)len;

        // Step 1: z = x * mult
        AscendC::Muls(zLocal, xLocal, mult, n);

        // Step 2: LeakyReLU (negative_slope=0.01): y = max(z, 0.01*z)
        AscendC::Muls(tmp1, zLocal, 0.01f, n);
        AscendC::Max(zLocal, zLocal, tmp1, n);

        // Step 3: GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        AscendC::Muls(tmp1, zLocal, 0.7071067811865475f, n);
        AscendC::Erf(tmp2, tmp1, n);
        AscendC::Adds(tmp2, tmp2, 1.0f, n);
        AscendC::Muls(tmp2, tmp2, 0.5f, n);
        AscendC::Mul(zLocal, zLocal, tmp2, n);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t c, uint32_t offset, uint32_t len) {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        uint64_t gmOffset = (uint64_t)c * (uint64_t)this->channelSize + (uint64_t)offset;
        AscendC::DataCopyPad(zGm[gmOffset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> multQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    AscendC::GlobalTensor<float> multiplierGm;
    uint32_t startChannel;
    uint32_t blockChannels;
    uint32_t channelSize;
    uint32_t channels;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_multiply_leaky_relu_gelu_custom(
    GM_ADDR x, GM_ADDR multiplier, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMultiplyLeakyReluGelu op;
    op.Init(x, multiplier, z,
            tiling_data.totalChannels,
            tiling_data.channelSize,
            tiling_data.channels,
            tiling_data.tileLength);
    op.Process();
}
