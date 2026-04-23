
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
                                 uint32_t B, uint32_t C, uint32_t D, uint32_t H, uint32_t W)
    {
        this->B = B; this->C = C; this->D = D; this->H = H; this->W = W;

        uint32_t totalBH = B * H;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t perBlock = (totalBH + blockNum - 1) / blockNum;
        uint32_t bhStart = blockIdx * perBlock;
        uint32_t bhEnd = bhStart + perBlock;
        if (bhEnd > totalBH) bhEnd = totalBH;
        this->bhStart = bhStart;
        this->bhCount = (bhEnd > bhStart) ? (bhEnd - bhStart) : 0;

        uint64_t totalElems = (uint64_t)B * C * D * H * W;
        uint64_t outElems = (uint64_t)B * C * H * W;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, C);
        yGm.SetGlobalBuffer((__gm__ float*)y, outElems);

        uint32_t alignedC = ((C * sizeof(float) + 31) / 32) * 32;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, W * sizeof(float));
        pipe.InitBuffer(meanBuf, C * W * sizeof(float));
        pipe.InitBuffer(biasBuf, alignedC);
        pipe.InitBuffer(rowMaxBuf, W * sizeof(float));
        pipe.InitBuffer(rowSumBuf, W * sizeof(float));
        pipe.InitBuffer(tmpBufU8, 32 * 1024);
    }

    __aicore__ inline void Process()
    {
        if (bhCount == 0) return;

        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopy(biasLocal, biasGm, C);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t i = 0; i < bhCount; i++) {
            uint32_t bh = bhStart + i;
            uint32_t b = bh / H;
            uint32_t h = bh % H;
            ProcessBH(b, h, biasLocal);
        }
    }

private:
    __aicore__ inline void ProcessBH(uint32_t b, uint32_t h, AscendC::LocalTensor<float>& biasLocal)
    {
        AscendC::LocalTensor<float> meanLocal = meanBuf.Get<float>();
        AscendC::LocalTensor<float> rowMax = rowMaxBuf.Get<float>();
        AscendC::LocalTensor<float> rowSum = rowSumBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> tmpU8 = tmpBufU8.Get<uint8_t>();

        uint32_t totalCW = C * W;
        AscendC::Duplicate<float>(meanLocal, 0.0f, totalCW);

        uint64_t xBase = (uint64_t)b * C * D * H * W + (uint64_t)h * W;
        uint64_t strideDHW = (uint64_t)D * H * W;
        uint64_t strideHW = (uint64_t)H * W;

        for (uint32_t d = 0; d < D; d++) {
            for (uint32_t c = 0; c < C; c++) {
                uint64_t offset = xBase + (uint64_t)c * strideDHW + (uint64_t)d * strideHW;
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[offset], W);
                inQueueX.EnQue(xLocal);
                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::Add<float>(meanLocal[c * W], meanLocal[c * W], xIn, W);
                inQueueX.FreeTensor(xIn);
            }
        }

        float invD = 1.0f / (float)D;
        AscendC::Muls<float>(meanLocal, meanLocal, invD, totalCW);
        for (uint32_t c = 0; c < C; c++) {
            float bc = biasLocal.GetValue(c);
            AscendC::Adds<float>(meanLocal[c * W], meanLocal[c * W], bc, W);
        }

        uint32_t srcShape[2] = {C, W};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(
            rowMax, meanLocal, tmpU8, srcShape, true);

        uint8_t repStride = (uint8_t)(W / 8);
        for (uint32_t wOff = 0; wOff < W; wOff += 64) {
            uint32_t curMask = (W - wOff >= 64) ? 64 : (W - wOff);
            AscendC::Sub<float>(meanLocal[wOff], meanLocal[wOff], rowMax[wOff],
                                (uint64_t)curMask, (uint8_t)C,
                                {1, 1, 1, repStride, repStride, 0});
        }

        AscendC::Exp<float>(meanLocal, meanLocal, totalCW);

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
            rowSum, meanLocal, tmpU8, srcShape, true);

        for (uint32_t wOff = 0; wOff < W; wOff += 64) {
            uint32_t curMask = (W - wOff >= 64) ? 64 : (W - wOff);
            AscendC::Div<float>(meanLocal[wOff], meanLocal[wOff], rowSum[wOff],
                                (uint64_t)curMask, (uint8_t)C,
                                {1, 1, 1, repStride, repStride, 0});
        }

        AscendC::Tanh<float>(meanLocal, meanLocal, totalCW);

        AscendC::PipeBarrier<PIPE_ALL>();

        uint64_t yBase = (uint64_t)b * C * H * W + (uint64_t)h * W;
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = (uint16_t)C;
        copyOutParams.blockLen = (uint32_t)(W * sizeof(float));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = (uint32_t)((H - 1) * W * sizeof(float));
        copyOutParams.rsv = 0;
        AscendC::DataCopyPad(yGm[yBase], meanLocal, copyOutParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> meanBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowSumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufU8;
    AscendC::GlobalTensor<float> xGm, biasGm, yGm;
    uint32_t B, C, D, H, W;
    uint32_t bhStart, bhCount;
};

extern "C" __global__ __aicore__ void convtranspose3d_mean_add_softmax_tanh_scaling_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, y, tiling_data.B, tiling_data.C, tiling_data.D, tiling_data.H, tiling_data.W);
    op.Process();
}
