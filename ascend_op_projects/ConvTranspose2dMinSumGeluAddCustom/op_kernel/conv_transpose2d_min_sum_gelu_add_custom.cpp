
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_H = 16;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t totalN, uint32_t totalC,
                                 uint32_t totalH, uint32_t totalW)
    {
        this->N = totalN;
        this->C = totalC;
        this->H = totalH;
        this->W = totalW;
        this->tileH = TILE_H;
        this->numHTiles = (this->H + this->tileH - 1) / this->tileH;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->startBatch = blockIdx;
        if (this->startBatch >= this->N) {
            this->endBatch = this->startBatch;
        } else {
            this->endBatch = this->startBatch + 1;
            if (this->endBatch > this->N) this->endBatch = this->N;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, (uint64_t)this->N * this->C * this->H * this->W);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, 1);
        yGm.SetGlobalBuffer((__gm__ float *)y, (uint64_t)this->N * this->W);

        uint32_t tileSize = this->tileH * this->W;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->W * sizeof(float));
        pipe.InitBuffer(biasQueue, 1, 32);
        pipe.InitBuffer(minBuf, tileSize * sizeof(float));
        pipe.InitBuffer(sumBuf, this->W * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->startBatch >= this->N) return;

        AscendC::LocalTensor<float> biasLocal = biasQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams biasCopyParams{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> biasPadParams{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, biasPadParams);
        biasQueue.EnQue(biasLocal);
        AscendC::LocalTensor<float> biasIn = biasQueue.DeQue<float>();
        float biasVal = biasIn.GetValue(0);
        biasQueue.FreeTensor(biasIn);

        for (uint32_t n = this->startBatch; n < this->endBatch; n++) {
            ProcessBatch(n, biasVal);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t n, float biasVal)
    {
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::Duplicate<float>(sumLocal, 0.0f, this->W);

        AscendC::LocalTensor<float> minLocal = minBuf.Get<float>();

        uint64_t batchOffset = (uint64_t)n * this->C * this->H * this->W;
        uint64_t chanStride = (uint64_t)this->H * this->W;

        for (uint32_t ht = 0; ht < this->numHTiles; ht++) {
            uint32_t hStart = ht * this->tileH;
            uint32_t rowsThisTile = this->tileH;
            if (hStart + rowsThisTile > this->H) rowsThisTile = this->H - hStart;
            uint32_t tileElems = rowsThisTile * this->W;

            AscendC::Duplicate<float>(minLocal, 1.0e30f, tileElems);

            for (uint32_t c = 0; c < this->C; c++) {
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                uint64_t xOffset = batchOffset + (uint64_t)c * chanStride + (uint64_t)hStart * this->W;
                AscendC::DataCopyExtParams copyParams{1, (uint32_t)(tileElems * sizeof(float)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                AscendC::DataCopyPad(xLocal, xGm[xOffset], copyParams, padParams);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::Min<float>(minLocal, minLocal, xIn, tileElems);
                inQueueX.FreeTensor(xIn);
            }

            for (uint32_t r = 0; r < rowsThisTile; r++) {
                AscendC::Add<float>(sumLocal, sumLocal, minLocal[r * this->W], this->W);
            }
        }

        AscendC::Muls<float>(minLocal, sumLocal, 0.7071067811865475f, this->W);
        AscendC::Erf<float>(minLocal, minLocal, this->W);
        AscendC::Adds<float>(minLocal, minLocal, 1.0f, this->W);
        AscendC::Mul<float>(minLocal, minLocal, sumLocal, this->W);

        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Muls<float>(yLocal, minLocal, 0.5f, this->W);
        AscendC::Adds<float>(yLocal, yLocal, biasVal, this->W);
        outQueueY.EnQue(yLocal);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams yCopyParams{1, (uint32_t)(this->W * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[(uint64_t)n * this->W], yOut, yCopyParams);
        outQueueY.FreeTensor(yOut);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> minBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t N, C, H, W;
    uint32_t tileH;
    uint32_t numHTiles;
    uint32_t startBatch, endBatch;
};

extern "C" __global__ __aicore__ void conv_transpose2d_min_sum_gelu_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, y, tiling_data.totalN, tiling_data.totalC, tiling_data.totalH, tiling_data.totalW);
    op.Process();
}
