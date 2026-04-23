
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFusedLeakyMulLeaky {
public:
    __aicore__ inline KernelFusedLeakyMulLeaky() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mul, GM_ADDR y,
                                uint32_t totalChannels, uint32_t channelSize,
                                uint32_t channels, uint32_t tileLen)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t channelsPerCore = (totalChannels + blockNum - 1) / blockNum;
        uint32_t startCh = blockIdx * channelsPerCore;
        uint32_t endCh = startCh + channelsPerCore;
        if (endCh > totalChannels) endCh = totalChannels;

        this->myChannels = (endCh > startCh) ? (endCh - startCh) : 0;
        this->startChannel = startCh;
        this->channels = channels;
        this->channelSize = channelSize;
        this->tileLen = tileLen;

        xGm.SetGlobalBuffer((__gm__ float *)x + (uint64_t)startCh * channelSize,
                            (uint64_t)this->myChannels * channelSize);
        yGm.SetGlobalBuffer((__gm__ float *)y + (uint64_t)startCh * channelSize,
                            (uint64_t)this->myChannels * channelSize);
        mGm.SetGlobalBuffer((__gm__ float *)mul, channels);

        pipe.InitBuffer(inQueue, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileLen * sizeof(float));

        uint32_t alignedC = (channels + 7) / 8 * 8;
        if (alignedC < 8) alignedC = 8;
        pipe.InitBuffer(mulQue, 1, alignedC * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (myChannels == 0) return;

        AscendC::LocalTensor<float> mulLocal = mulQue.AllocTensor<float>();
        AscendC::DataCopyExtParams copyMul{1, (uint32_t)(channels * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padMul{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(mulLocal, mGm, copyMul, padMul);
        mulQue.EnQue(mulLocal);
        mulLocal = mulQue.DeQue<float>();

        const float negSlope = 0.2f;

        for (uint32_t c = 0; c < myChannels; c++) {
            uint32_t globalCh = startChannel + c;
            uint32_t mulIdx = globalCh % channels;
            float m = mulLocal.GetValue(mulIdx);

            uint64_t baseOffset = (uint64_t)c * channelSize;
            uint32_t numTiles = (channelSize + tileLen - 1) / tileLen;

            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t tileOffset = t * tileLen;
                uint32_t curLen = (tileOffset + tileLen <= channelSize) ?
                                  tileLen : (channelSize - tileOffset);
                ProcessTile(baseOffset + tileOffset, curLen, m, negSlope);
            }
        }

        mulQue.FreeTensor(mulLocal);
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t len, float m, float negSlope)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams copyIn{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padIn{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyIn, padIn);
        inQueue.EnQue(xLocal);

        xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        AscendC::LeakyRelu<float>(yLocal, xLocal, negSlope, len);
        AscendC::Muls<float>(yLocal, yLocal, m, len);
        AscendC::LeakyRelu<float>(yLocal, yLocal, negSlope, len);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);

        yLocal = outQueue.DeQue<float>();
        AscendC::DataCopyExtParams copyOut{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyOut);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> mulQue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> mGm;
    uint32_t myChannels;
    uint32_t startChannel;
    uint32_t channels;
    uint32_t channelSize;
    uint32_t tileLen;
};

extern "C" __global__ __aicore__ void conv_transpose3d_leaky_relu_multiply_leaky_relu_max_custom(
    GM_ADDR x, GM_ADDR multiplier, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedLeakyMulLeaky op;
    op.Init(x, multiplier, y,
            tiling_data.totalChannels,
            tiling_data.channelSize,
            tiling_data.channels,
            tiling_data.tileLen);
    op.Process();
}
