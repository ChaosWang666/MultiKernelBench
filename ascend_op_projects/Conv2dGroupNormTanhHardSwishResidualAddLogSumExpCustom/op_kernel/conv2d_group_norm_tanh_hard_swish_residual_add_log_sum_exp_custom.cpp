
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t CHANNELS = 64;
constexpr int32_t ROWS_PER_TILE = 32;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}

    __aicore__ inline void Init(GM_ADDR xNorm, GM_ADDR xConv, GM_ADDR out,
                                uint32_t totalPositions)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t base = totalPositions / blockNum;
        uint32_t rem  = totalPositions % blockNum;

        uint32_t startPos;
        uint32_t myPositions;
        if (blockIdx < rem) {
            myPositions = base + 1;
            startPos = blockIdx * myPositions;
        } else {
            myPositions = base;
            startPos = rem * (base + 1) + (blockIdx - rem) * base;
        }

        this->myPositions = myPositions;
        this->tileCount = (myPositions + ROWS_PER_TILE - 1) / ROWS_PER_TILE;

        xNormGm.SetGlobalBuffer((__gm__ float *)xNorm + startPos * CHANNELS, myPositions * CHANNELS);
        xConvGm.SetGlobalBuffer((__gm__ float *)xConv + startPos * CHANNELS, myPositions * CHANNELS);
        outGm.SetGlobalBuffer((__gm__ float *)out + startPos, myPositions);

        pipe.InitBuffer(inQueueXNorm, BUFFER_NUM, ROWS_PER_TILE * CHANNELS * sizeof(float));
        pipe.InitBuffer(inQueueXConv, BUFFER_NUM, ROWS_PER_TILE * CHANNELS * sizeof(float));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, ROWS_PER_TILE * sizeof(float));
        pipe.InitBuffer(tmpBufA, ROWS_PER_TILE * CHANNELS * sizeof(float));
        pipe.InitBuffer(tmpBufB, ROWS_PER_TILE * CHANNELS * sizeof(float));
        pipe.InitBuffer(reduceBuf, 1024);
        pipe.InitBuffer(scalarBuf, 32);
    }

    __aicore__ inline void Process()
    {
        if (this->myPositions == 0) {
            return;
        }
        for (uint32_t t = 0; t < this->tileCount; t++) {
            uint32_t startRow = t * ROWS_PER_TILE;
            uint32_t rowsThisTile = ROWS_PER_TILE;
            if (startRow + ROWS_PER_TILE > this->myPositions) {
                rowsThisTile = this->myPositions - startRow;
            }
            CopyIn(t, rowsThisTile);
            Compute(t, rowsThisTile);
            CopyOut(t, rowsThisTile);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t rows)
    {
        AscendC::LocalTensor<float> xNormLocal = inQueueXNorm.AllocTensor<float>();
        AscendC::LocalTensor<float> xConvLocal = inQueueXConv.AllocTensor<float>();

        uint32_t gmOffset = progress * ROWS_PER_TILE * CHANNELS;
        uint32_t count = rows * CHANNELS;

        AscendC::DataCopy(xNormLocal, xNormGm[gmOffset], count);
        AscendC::DataCopy(xConvLocal, xConvGm[gmOffset], count);

        inQueueXNorm.EnQue(xNormLocal);
        inQueueXConv.EnQue(xConvLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t rows)
    {
        AscendC::LocalTensor<float> xNormLocal = inQueueXNorm.DeQue<float>();
        AscendC::LocalTensor<float> xConvLocal = inQueueXConv.DeQue<float>();
        AscendC::LocalTensor<float> outLocal  = outQueueOut.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpA = tmpBufA.Get<float>();
        AscendC::LocalTensor<float> tmpB = tmpBufB.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        uint32_t count = rows * CHANNELS;

        // 1. tanh(x_norm) -> tmpA
        AscendC::Tanh<float>(tmpA, xNormLocal, count);

        // 2. HardSwish(tanh):
        //    tanh in [-1, 1]  => (tanh + 3) in [2, 4] => relu6(tanh + 3) = tanh + 3
        //    hardswish(tanh) = tanh * (tanh + 3) / 6
        AscendC::Adds<float>(tmpB, tmpA, 3.0f, count);
        AscendC::Mul<float>(tmpB, tmpB, tmpA, count);
        AscendC::Muls<float>(tmpB, tmpB, 1.0f / 6.0f, count);

        // 3. residual = x_conv + hardswish -> tmpA
        AscendC::Add<float>(tmpA, xConvLocal, tmpB, count);

        // 4. LogSumExp per row (reduce along CHANNELS)
        for (uint32_t r = 0; r < rows; r++) {
            uint32_t rowOff = r * CHANNELS;

            AscendC::ReduceMax<float>(scalarLocal, tmpA[rowOff], reduceTmp,
                                       static_cast<int32_t>(CHANNELS), false);
            float maxVal = scalarLocal.GetValue(0);

            AscendC::Adds<float>(tmpB[rowOff], tmpA[rowOff], -maxVal,
                                  static_cast<int32_t>(CHANNELS));

            AscendC::Exp<float>(tmpB[rowOff], tmpB[rowOff],
                                 static_cast<int32_t>(CHANNELS));

            AscendC::ReduceSum<float, true>(scalarLocal, tmpB[rowOff], reduceTmp,
                                             static_cast<int32_t>(CHANNELS));
            float sumVal = scalarLocal.GetValue(0);

            AscendC::Duplicate<float>(scalarLocal, sumVal, 8);
            AscendC::Ln<float>(scalarLocal, scalarLocal, 8);
            float logSum = scalarLocal.GetValue(0);

            outLocal.SetValue(r, maxVal + logSum);
        }

        outQueueOut.EnQue<float>(outLocal);
        inQueueXNorm.FreeTensor(xNormLocal);
        inQueueXConv.FreeTensor(xConvLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t rows)
    {
        AscendC::LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        uint32_t gmOffset = progress * ROWS_PER_TILE;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rows * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPad(outGm[gmOffset], outLocal, copyParams);

        outQueueOut.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueXNorm;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueXConv;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufA;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBufB;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xNormGm;
    AscendC::GlobalTensor<float> xConvGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t myPositions;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void conv2d_group_norm_tanh_hard_swish_residual_add_log_sum_exp_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, y, z, tiling_data.totalPositions);
    op.Process();
}
