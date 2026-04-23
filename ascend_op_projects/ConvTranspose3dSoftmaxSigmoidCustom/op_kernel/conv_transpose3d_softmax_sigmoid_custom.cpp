
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConvTransSoftmaxSigmoid {
public:
    __aicore__ inline KernelConvTransSoftmaxSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t cols,
                                uint32_t rowsPerCore, uint32_t rowsPerTile)
    {
        this->cols = cols;
        this->rowsPerTile = rowsPerTile;
        this->colsAlign = (cols + 7) / 8 * 8;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerCore;
        uint32_t endRow = startRow + rowsPerCore;
        if (endRow > totalRows) endRow = totalRows;
        this->myRows = (endRow > startRow) ? (endRow - startRow) : 0;

        if (this->myRows == 0) return;

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * cols, this->myRows * cols);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * cols, this->myRows * cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowsPerTile * colsAlign * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, rowsPerTile * colsAlign * sizeof(float));
        pipe.InitBuffer(scalarBuf, 32);
        pipe.InitBuffer(reduceBuf, 1024);
        pipe.InitBuffer(onesBuf, colsAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (myRows == 0) return;

        AscendC::LocalTensor<float> onesLocal = onesBuf.Get<float>();
        AscendC::Duplicate<float>(onesLocal, 1.0f, colsAlign);

        uint32_t numTiles = (myRows + rowsPerTile - 1) / rowsPerTile;
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t rowStart = t * rowsPerTile;
            uint32_t rowsThis = (myRows - rowStart < rowsPerTile) ? (myRows - rowStart) : rowsPerTile;
            CopyIn(rowStart, rowsThis);
            Compute(rowsThis);
            CopyOut(rowStart, rowsThis);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowStart, uint32_t rowsThis)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowsThis);
        copyParams.blockLen = cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;

        AscendC::DataCopyPad(xLocal, xGm[rowStart * cols], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowsThis)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> onesLocal = onesBuf.Get<float>();

        for (uint32_t r = 0; r < rowsThis; r++) {
            uint32_t rowOffset = r * colsAlign;

            // Softmax: max
            AscendC::ReduceMax<float>(scalarLocal, xLocal[rowOffset], reduceTmp,
                                      static_cast<int32_t>(cols), false);
            float maxVal = scalarLocal.GetValue(0);

            // x - max
            AscendC::Adds<float>(yLocal[rowOffset], xLocal[rowOffset], -maxVal, cols);

            // exp(x - max)
            AscendC::Exp<float>(yLocal[rowOffset], yLocal[rowOffset], cols);

            // sum(exp)
            AscendC::ReduceSum<float, true>(scalarLocal, yLocal[rowOffset], reduceTmp,
                                            static_cast<int32_t>(cols));
            float sumVal = scalarLocal.GetValue(0);
            float invSum = 1.0f / sumVal;

            // softmax = exp / sum
            AscendC::Muls<float>(yLocal[rowOffset], yLocal[rowOffset], invSum, cols);

            // Sigmoid: 1 / (1 + exp(-y))
            AscendC::Muls<float>(yLocal[rowOffset], yLocal[rowOffset], -1.0f, cols);
            AscendC::Exp<float>(yLocal[rowOffset], yLocal[rowOffset], cols);
            AscendC::Adds<float>(yLocal[rowOffset], yLocal[rowOffset], 1.0f, cols);
            AscendC::Div<float>(yLocal[rowOffset], onesLocal, yLocal[rowOffset], cols);
        }

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowStart, uint32_t rowsThis)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowsThis);
        copyParams.blockLen = cols * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(yGm[rowStart * cols], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t cols;
    uint32_t colsAlign;
    uint32_t myRows;
    uint32_t rowsPerTile;
};

extern "C" __global__ __aicore__ void conv_transpose3d_softmax_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransSoftmaxSigmoid op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.cols,
            tiling_data.rowsPerCore, tiling_data.rowsPerTile);
    op.Process();
}
