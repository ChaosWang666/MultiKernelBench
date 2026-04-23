
#include "kernel_operator.h"
#include <math.h>

constexpr int32_t BUFFER_NUM = 1;

class KernelGemmGroupNormHardtanh {
public:
    __aicore__ inline KernelGemmGroupNormHardtanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                 uint32_t totalRows, uint32_t outFeatures,
                                 uint32_t numGroups, uint32_t rowsPerBlock,
                                 float htMin, float htMax, float eps)
    {
        this->outFeatures = outFeatures;
        this->numGroups = numGroups;
        this->groupSize = outFeatures / numGroups;
        this->invGroupSize = 1.0f / (float)this->groupSize;
        this->htMin = htMin;
        this->htMax = htMax;
        this->eps = eps;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerBlock;
        if (startRow >= totalRows) {
            this->numRowsThisBlock = 0;
            return;
        }
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > totalRows) endRow = totalRows;
        this->numRowsThisBlock = endRow - startRow;

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * outFeatures,
                            this->numRowsThisBlock * outFeatures);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * outFeatures,
                            this->numRowsThisBlock * outFeatures);
        gammaGm.SetGlobalBuffer((__gm__ float*)gamma, outFeatures);
        betaGm.SetGlobalBuffer((__gm__ float*)beta, outFeatures);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, outFeatures * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outFeatures * sizeof(float));
        pipe.InitBuffer(gammaBuf, outFeatures * sizeof(float));
        pipe.InitBuffer(betaBuf, outFeatures * sizeof(float));
        pipe.InitBuffer(reduceBuf, 1024);
        pipe.InitBuffer(scalarBuf, 64);
    }

    __aicore__ inline void Process()
    {
        if (numRowsThisBlock == 0) return;

        AscendC::LocalTensor<float> gammaLocal = gammaBuf.Get<float>();
        AscendC::LocalTensor<float> betaLocal = betaBuf.Get<float>();
        AscendC::DataCopy(gammaLocal, gammaGm, outFeatures);
        AscendC::DataCopy(betaLocal, betaGm, outFeatures);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t row = 0; row < numRowsThisBlock; row++) {
            CopyIn(row);
            Compute(row, gammaLocal, betaLocal);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * outFeatures], outFeatures);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowIdx,
                                    AscendC::LocalTensor<float>& gammaLocal,
                                    AscendC::LocalTensor<float>& betaLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        for (uint32_t g = 0; g < numGroups; g++) {
            uint32_t offset = g * groupSize;

            AscendC::ReduceSum<float, true>(scalarLocal, xLocal[offset], reduceTmp,
                                             (int32_t)groupSize);
            float sumVal = scalarLocal.GetValue(0);
            float mean = sumVal * invGroupSize;

            AscendC::Adds<float>(yLocal[offset], xLocal[offset], -mean, groupSize);

            AscendC::Mul<float>(xLocal[offset], yLocal[offset], yLocal[offset], groupSize);
            AscendC::ReduceSum<float, true>(scalarLocal, xLocal[offset], reduceTmp,
                                             (int32_t)groupSize);
            float varSum = scalarLocal.GetValue(0);
            float var = varSum * invGroupSize;
            float stdVal = sqrtf(var + eps);
            float invStd = 1.0f / stdVal;

            AscendC::Muls<float>(yLocal[offset], yLocal[offset], invStd, groupSize);
        }

        inQueueX.FreeTensor(xLocal);

        AscendC::Mul<float>(yLocal, yLocal, gammaLocal, outFeatures);
        AscendC::Add<float>(yLocal, yLocal, betaLocal, outFeatures);

        AscendC::Mins<float>(yLocal, yLocal, htMax, outFeatures);
        AscendC::Maxs<float>(yLocal, yLocal, htMin, outFeatures);

        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[rowIdx * outFeatures], yLocal, outFeatures);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf, betaBuf, reduceBuf, scalarBuf;
    AscendC::GlobalTensor<float> xGm, yGm, gammaGm, betaGm;
    uint32_t numRowsThisBlock;
    uint32_t outFeatures, numGroups, groupSize;
    float invGroupSize, htMin, htMax, eps;
};

extern "C" __global__ __aicore__ void gemm_group_norm_hardtanh_custom(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmGroupNormHardtanh op;
    op.Init(x, gamma, beta, y,
            tiling_data.totalRows, tiling_data.outFeatures,
            tiling_data.numGroups, tiling_data.rowsPerBlock,
            tiling_data.htMin, tiling_data.htMax, tiling_data.eps);
    op.Process();
}
