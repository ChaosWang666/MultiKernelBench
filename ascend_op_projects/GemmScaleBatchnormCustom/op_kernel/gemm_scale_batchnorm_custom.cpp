
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmScaleBatchnorm {
public:
    __aicore__ inline KernelGemmScaleBatchnorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR a, GM_ADDR b, GM_ADDR y,
                                 uint32_t batchSize, uint32_t features,
                                 uint32_t rowsPerCore, uint32_t tileRows)
    {
        this->features = features;
        this->tileRows = tileRows;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > batchSize) {
            endRow = batchSize;
        }
        this->myRows = (endRow > startRow) ? (endRow - startRow) : 0;

        if (this->myRows == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * features, this->myRows * features);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * features, this->myRows * features);
        aGm.SetGlobalBuffer((__gm__ float*)a, features);
        bGm.SetGlobalBuffer((__gm__ float*)b, features);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileRows * features * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileRows * features * sizeof(float));
        pipe.InitBuffer(aBuf, features * sizeof(float));
        pipe.InitBuffer(bBuf, features * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0) {
            return;
        }

        AscendC::LocalTensor<float> aLocal = aBuf.Get<float>();
        AscendC::LocalTensor<float> bLocal = bBuf.Get<float>();
        AscendC::DataCopy(aLocal, aGm, this->features);
        AscendC::DataCopy(bLocal, bGm, this->features);
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t numTiles = (this->myRows + this->tileRows - 1) / this->tileRows;
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t tileStartRow = t * this->tileRows;
            uint32_t remaining = this->myRows - tileStartRow;
            uint32_t rowsThisTile = (remaining > this->tileRows) ? this->tileRows : remaining;

            CopyIn(tileStartRow, rowsThisTile);
            Compute(rowsThisTile, aLocal, bLocal);
            CopyOut(tileStartRow, rowsThisTile);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t tileStartRow, uint32_t rowsThisTile)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[tileStartRow * this->features], rowsThisTile * this->features);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowsThisTile,
                                     AscendC::LocalTensor<float>& aLocal,
                                     AscendC::LocalTensor<float>& bLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        for (uint32_t r = 0; r < rowsThisTile; r++) {
            uint32_t offset = r * this->features;
            AscendC::Mul<float>(yLocal[offset], xLocal[offset], aLocal, this->features);
            AscendC::Add<float>(yLocal[offset], yLocal[offset], bLocal, this->features);
        }

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileStartRow, uint32_t rowsThisTile)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[tileStartRow * this->features], yLocal, rowsThisTile * this->features);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t features;
    uint32_t tileRows;
    uint32_t myRows;
};

extern "C" __global__ __aicore__ void gemm_scale_batchnorm_custom(
    GM_ADDR x, GM_ADDR a, GM_ADDR b, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmScaleBatchnorm op;
    op.Init(x, a, b, y,
            tiling_data.batchSize, tiling_data.features,
            tiling_data.rowsPerCore, tiling_data.tileRows);
    op.Process();
}
