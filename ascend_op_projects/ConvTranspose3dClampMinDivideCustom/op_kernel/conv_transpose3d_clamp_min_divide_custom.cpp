
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConvTranspose3dClampMinDivide {
public:
    __aicore__ inline KernelConvTranspose3dClampMinDivide() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum,
                                 float minValue, float invDivisor)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t alignedTotal = (totalLength + 7) / 8 * 8;
        uint32_t baseBlockLen = alignedTotal / blockNum;
        baseBlockLen = (baseBlockLen + 7) / 8 * 8;

        uint32_t startOffset = blockIdx * baseBlockLen;
        uint32_t curBlockLen = 0;
        if (startOffset < totalLength) {
            uint32_t remain = totalLength - startOffset;
            curBlockLen = (remain < baseBlockLen) ? remain : baseBlockLen;
        }

        this->blockLength = curBlockLen;
        this->startOffset = startOffset;
        this->minValue = minValue;
        this->invDivisor = invDivisor;

        if (this->blockLength == 0) {
            this->tileNum = 1;
            this->tileLength = 0;
            this->lastTileLength = 0;
            return;
        }

        uint32_t tn = tileNum;
        if (tn == 0) tn = 1;
        uint32_t tl = (this->blockLength + tn * BUFFER_NUM - 1) / (tn * BUFFER_NUM);
        tl = (tl + 7) / 8 * 8;
        if (tl == 0) tl = 8;

        uint32_t totalLoops = (this->blockLength + tl - 1) / tl;
        this->tileNum = totalLoops;
        this->tileLength = tl;
        this->lastTileLength = this->blockLength - (totalLoops - 1) * tl;

        xGm.SetGlobalBuffer((__gm__ float *)x + startOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + startOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;
        uint32_t loopCount = this->tileNum;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLen = (i == loopCount - 1) ? this->lastTileLength : this->tileLength;
            CopyIn(i, curLen);
            Compute(i, curLen);
            CopyOut(i, curLen);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curLen)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress, uint32_t curLen)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        uint32_t alignLen = (curLen + 7) / 8 * 8;
        AscendC::Maxs<float>(yLocal, xLocal, this->minValue, alignLen);
        AscendC::Muls<float>(yLocal, yLocal, this->invDivisor, alignLen);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curLen)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[progress * this->tileLength], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t startOffset;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    float minValue;
    float invDivisor;
};

extern "C" __global__ __aicore__ void conv_transpose3d_clamp_min_divide_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dClampMinDivide op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum,
            tiling_data.minValue, tiling_data.invDivisor);
    op.Process();
}
