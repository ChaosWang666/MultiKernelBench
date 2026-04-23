
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelChannelSum {
public:
    __aicore__ inline KernelChannelSum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t B, uint32_t C, uint32_t DHW,
                                uint32_t tileLen, uint32_t tilesPerBatch,
                                uint32_t totalTiles, uint32_t tilesPerCore)
    {
        this->B = B;
        this->C = C;
        this->DHW = DHW;
        this->tileLen = tileLen;
        this->tilesPerBatch = tilesPerBatch;
        this->totalTiles = totalTiles;
        this->tilesPerCore = tilesPerCore;

        uint64_t inSize = (uint64_t)B * C * DHW;
        uint64_t outSize = (uint64_t)B * DHW;

        xGm.SetGlobalBuffer((__gm__ float*)x, inSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, outSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t startTile = coreIdx * tilesPerCore;
        uint32_t endTile = startTile + tilesPerCore;
        if (endTile > totalTiles) endTile = totalTiles;

        for (uint32_t tileIdx = startTile; tileIdx < endTile; tileIdx++) {
            uint32_t b = tileIdx / tilesPerBatch;
            uint32_t t = tileIdx % tilesPerBatch;
            uint32_t pStart = t * tileLen;
            uint32_t actualLen = tileLen;
            if (pStart >= DHW) continue;
            if (pStart + actualLen > DHW) actualLen = DHW - pStart;
            if (actualLen == 0) continue;

            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            AscendC::Duplicate<float>(yLocal, 0.0f, tileLen);

            uint64_t inBaseOffset = (uint64_t)b * C * DHW + pStart;

            for (uint32_t c = 0; c < C; c++) {
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

                AscendC::DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = actualLen * sizeof(float);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = 0;
                padParams.paddingValue = 0.0f;

                AscendC::DataCopyPad(xLocal, xGm[inBaseOffset + (uint64_t)c * DHW], copyParams, padParams);

                inQueueX.EnQue(xLocal);
                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();

                AscendC::Add<float>(yLocal, yLocal, xIn, actualLen);

                inQueueX.FreeTensor(xIn);
            }

            outQueueY.EnQue(yLocal);
            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();

            uint64_t outOffset = (uint64_t)b * DHW + pStart;
            AscendC::DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = actualLen * sizeof(float);
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            AscendC::DataCopyPad(yGm[outOffset], yOut, outParams);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t B;
    uint32_t C;
    uint32_t DHW;
    uint32_t tileLen;
    uint32_t tilesPerBatch;
    uint32_t totalTiles;
    uint32_t tilesPerCore;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_max_sum_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelChannelSum op;
    op.Init(x, y,
            tiling_data.B, tiling_data.C, tiling_data.DHW,
            tiling_data.tileLen, tiling_data.tilesPerBatch,
            tiling_data.totalTiles, tiling_data.tilesPerCore);
    op.Process();
}
