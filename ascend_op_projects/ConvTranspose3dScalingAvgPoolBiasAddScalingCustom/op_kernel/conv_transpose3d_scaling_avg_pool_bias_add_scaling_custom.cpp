
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelPostProcess {
public:
    __aicore__ inline KernelPostProcess() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR y,
                                uint32_t totalOuter, uint32_t numChannels, uint32_t spatial, uint32_t tileSize)
    {
        this->numChannels = numChannels;
        this->spatial = spatial;
        this->tileSize = tileSize;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t outerPerBlock = (totalOuter + blockNum - 1) / blockNum;
        this->outerStart = blockIdx * outerPerBlock;
        uint32_t outerEndTmp = this->outerStart + outerPerBlock;
        this->outerEnd = (outerEndTmp > totalOuter) ? totalOuter : outerEndTmp;

        uint64_t totalElements = (uint64_t)totalOuter * (uint64_t)spatial;
        xGm.SetGlobalBuffer((__gm__ float *)x, totalElements);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalElements);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, numChannels);
        scale1Gm.SetGlobalBuffer((__gm__ float *)scale1, 1);
        scale2Gm.SetGlobalBuffer((__gm__ float *)scale2, 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(scale1Buf, 32);
        pipe.InitBuffer(scale2Buf, 32);

        uint32_t biasAlignedSize = ((numChannels * sizeof(float) + 31) / 32) * 32;
        if (biasAlignedSize < 32) {
            biasAlignedSize = 32;
        }
        pipe.InitBuffer(biasBuf, biasAlignedSize);
    }

    __aicore__ inline void Process()
    {
        if (outerStart >= outerEnd) {
            return;
        }

        AscendC::LocalTensor<float> scale1Local = scale1Buf.Get<float>();
        AscendC::LocalTensor<float> scale2Local = scale2Buf.Get<float>();
        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();

        {
            AscendC::DataCopyExtParams cp{1, (uint32_t)sizeof(float), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(scale1Local, scale1Gm, cp, pp);
            AscendC::DataCopyPad(scale2Local, scale2Gm, cp, pp);
        }
        {
            AscendC::DataCopyExtParams cp{1, (uint32_t)(numChannels * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(biasLocal, biasGm, cp, pp);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        float scale1Val = scale1Local.GetValue(0);
        float scale2Val = scale2Local.GetValue(0);
        float combinedScale = scale1Val * scale2Val;

        for (uint32_t outer = outerStart; outer < outerEnd; outer++) {
            uint32_t c = outer % numChannels;
            float biasVal = biasLocal.GetValue(c);
            float biasScaled = scale2Val * biasVal;

            uint64_t baseOffset = (uint64_t)outer * (uint64_t)spatial;
            uint32_t processed = 0;
            while (processed < spatial) {
                uint32_t remain = spatial - processed;
                uint32_t curSize = (remain < tileSize) ? remain : tileSize;
                CopyIn(baseOffset + processed, curSize);
                Compute(curSize, combinedScale, biasScaled);
                CopyOut(baseOffset + processed, curSize);
                processed += curSize;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t size)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(size * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[offset], cp, pp);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t size, float scale, float bias)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Muls<float>(yLocal, xLocal, scale, size);
        AscendC::Adds<float>(yLocal, yLocal, bias, size);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t size)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams cp{1, (uint32_t)(size * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, cp);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scale1Buf, scale2Buf, biasBuf;
    AscendC::GlobalTensor<float> xGm, yGm, biasGm, scale1Gm, scale2Gm;
    uint32_t numChannels;
    uint32_t spatial;
    uint32_t tileSize;
    uint32_t outerStart;
    uint32_t outerEnd;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scaling_avg_pool_bias_add_scaling_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelPostProcess op;
    op.Init(x, bias, scale1, scale2, y,
            tiling_data.totalOuter, tiling_data.numChannels, tiling_data.spatial, tiling_data.tileSize);
    op.Process();
}
