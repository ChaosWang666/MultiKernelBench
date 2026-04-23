
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dReluBiasAdd {
public:
    __aicore__ inline KernelConv2dReluBiasAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t totalGroups, uint32_t elementsPerGroup,
                                 uint32_t outChannels, uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();

        uint32_t groupsPerBlock = totalGroups / numBlocks;
        uint32_t remainder = totalGroups % numBlocks;

        uint32_t startG, endG;
        if (blockIdx < remainder) {
            startG = blockIdx * (groupsPerBlock + 1);
            endG = startG + groupsPerBlock + 1;
        } else {
            startG = blockIdx * groupsPerBlock + remainder;
            endG = startG + groupsPerBlock;
        }

        this->startGroup = startG;
        this->endGroup = endG;
        this->elementsPerGroup = elementsPerGroup;
        this->outChannels = outChannels;
        this->tileLength = tileLength;

        uint64_t totalElements = (uint64_t)totalGroups * (uint64_t)elementsPerGroup;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalElements);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElements);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(float));

        uint32_t biasAligned = (outChannels + 7) / 8 * 8;
        pipe.InitBuffer(biasBuf, biasAligned * sizeof(float));

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopyExtParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = outChannels * sizeof(float);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        biasCopyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> biasPadParams;
        biasPadParams.isPad = false;
        biasPadParams.leftPadding = 0;
        biasPadParams.rightPadding = 0;
        biasPadParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, biasPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();

        for (uint32_t g = startGroup; g < endGroup; g++) {
            uint32_t channelIdx = g % outChannels;
            float biasVal = biasLocal.GetValue(channelIdx);

            uint64_t gmOffset = (uint64_t)g * (uint64_t)elementsPerGroup;
            uint32_t numTiles = (elementsPerGroup + tileLength - 1) / tileLength;

            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t curLen = tileLength;
                if (t == numTiles - 1) {
                    curLen = elementsPerGroup - t * tileLength;
                }
                uint64_t offset = gmOffset + (uint64_t)t * (uint64_t)tileLength;
                CopyIn(offset, curLen);
                Compute(biasVal, curLen);
                CopyOut(offset, curLen);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(float biasVal, uint32_t len)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        AscendC::Relu<float>(yLocal, xLocal, len);
        AscendC::Adds<float>(yLocal, yLocal, biasVal, len);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t len)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(float);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t startGroup;
    uint32_t endGroup;
    uint32_t elementsPerGroup;
    uint32_t outChannels;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_relu_bias_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dReluBiasAdd op;
    op.Init(x, bias, y, tiling_data.totalGroups, tiling_data.elementsPerGroup,
            tiling_data.outChannels, tiling_data.tileLength);
    op.Process();
}
