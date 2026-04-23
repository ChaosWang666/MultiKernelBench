
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ROWS = 8;

class KernelLogSumExp {
public:
    __aicore__ inline KernelLogSumExp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t featureSize, uint32_t coreNum)
    {
        this->batchSize = batchSize;
        this->featureSize = featureSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t baseRows = batchSize / coreNum;
        uint32_t extraRows = batchSize % coreNum;
        uint32_t startRow;
        uint32_t endRow;
        if (blockIdx < extraRows) {
            startRow = blockIdx * (baseRows + 1);
            endRow = startRow + (baseRows + 1);
        } else {
            startRow = extraRows * (baseRows + 1) + (blockIdx - extraRows) * baseRows;
            endRow = startRow + baseRows;
        }

        this->rowsToProcess = endRow - startRow;
        if (this->rowsToProcess == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + startRow * featureSize,
                            this->rowsToProcess * featureSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + startRow, this->rowsToProcess);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_ROWS * featureSize * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
        pipe.InitBuffer(rowTmpBuf, featureSize * sizeof(DTYPE_X));
        pipe.InitBuffer(maxBuf, 32);
        pipe.InitBuffer(sumBuf, 32);
        pipe.InitBuffer(scalarBuf, 32);
        pipe.InitBuffer(reduceTmpBuf, 4096);
    }

    __aicore__ inline void Process()
    {
        if (rowsToProcess == 0) {
            return;
        }

        uint32_t tileCount = (rowsToProcess + TILE_ROWS - 1) / TILE_ROWS;
        for (uint32_t t = 0; t < tileCount; t++) {
            uint32_t startLocal = t * TILE_ROWS;
            uint32_t rowsThisTile = (startLocal + TILE_ROWS <= rowsToProcess)
                                        ? TILE_ROWS
                                        : (rowsToProcess - startLocal);

            CopyIn(startLocal, rowsThisTile);
            Compute(rowsThisTile);
            CopyOut(startLocal, rowsThisTile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t startLocal, uint32_t rowsThisTile)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowsThisTile * featureSize * sizeof(DTYPE_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DTYPE_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = (DTYPE_X)0;
        AscendC::DataCopyPad(xLocal, xGm[startLocal * featureSize], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowsThisTile)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> rowTmp = rowTmpBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> maxLocal = maxBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> sumLocal = sumBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> scalarTmp = scalarBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> reduceTmp = reduceTmpBuf.Get<DTYPE_X>();

        for (uint32_t r = 0; r < rowsThisTile; r++) {
            uint32_t offset = r * featureSize;

            AscendC::ReduceMax<DTYPE_X>(scalarTmp, xLocal[offset], reduceTmp,
                                        static_cast<int32_t>(featureSize), false);
            DTYPE_X maxVal = scalarTmp.GetValue(0);
            maxLocal.SetValue(r, maxVal);

            AscendC::Adds<DTYPE_X>(rowTmp, xLocal[offset], (DTYPE_X)(-maxVal), featureSize);

            AscendC::Exp<DTYPE_X>(rowTmp, rowTmp, featureSize);

            AscendC::ReduceSum<DTYPE_X, true>(scalarTmp, rowTmp, reduceTmp,
                                              static_cast<int32_t>(featureSize));
            DTYPE_X sumVal = scalarTmp.GetValue(0);
            sumLocal.SetValue(r, sumVal);
        }

        for (uint32_t r = rowsThisTile; r < TILE_ROWS; r++) {
            sumLocal.SetValue(r, (DTYPE_X)1.0);
            maxLocal.SetValue(r, (DTYPE_X)0.0);
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Log<DTYPE_X>(sumLocal, sumLocal, TILE_ROWS);
        AscendC::Add<DTYPE_X>(yLocal, maxLocal, sumLocal, TILE_ROWS);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t startLocal, uint32_t rowsThisTile)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rowsThisTile * sizeof(DTYPE_Y);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[startLocal], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t featureSize;
    uint32_t rowsToProcess;
};

extern "C" __global__ __aicore__ void gemm_sigmoid_sum_log_sum_exp_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelLogSumExp op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.featureSize, tiling_data.coreNum);
    op.Process();
}
