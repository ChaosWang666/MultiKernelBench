
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_SIZE = 4096;

class KernelMatmulDropoutMeanSoftmax {
public:
    __aicore__ inline KernelMatmulDropoutMeanSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t cols)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        uint32_t sRow = blockIdx * rowsPerBlock;
        uint32_t eRow = sRow + rowsPerBlock;
        if (eRow > totalRows) eRow = totalRows;
        if (sRow > totalRows) sRow = totalRows;

        this->cols = cols;
        this->startRow = sRow;
        this->endRow = eRow;

        xGm.SetGlobalBuffer((__gm__ float *)x, (uint64_t)totalRows * cols);
        yGm.SetGlobalBuffer((__gm__ float *)y, (uint64_t)totalRows * cols);

        pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_SIZE * sizeof(float));
        pipe.InitBuffer(reduceBuf, 2048);
        pipe.InitBuffer(scalarBuf, 32);
    }

    __aicore__ inline void Process()
    {
        if (startRow >= endRow) {
            return;
        }
        uint32_t numTiles = (cols + TILE_SIZE - 1) / TILE_SIZE;

        for (uint32_t row = startRow; row < endRow; row++) {
            float maxVal = -3.4e38f;

            // Pass 1: find max across the row
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t tileStart = t * TILE_SIZE;
                uint32_t tileLen = (tileStart + TILE_SIZE <= cols) ? TILE_SIZE : (cols - tileStart);

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[row * cols + tileStart], tileLen);
                inQueue.EnQue(xLocal);
                AscendC::LocalTensor<float> xIn = inQueue.DeQue<float>();

                AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
                AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
                AscendC::ReduceMax<float>(scalarLocal, xIn, reduceTmp, (int32_t)tileLen, false);
                float tileMax = scalarLocal.GetValue(0);
                if (tileMax > maxVal) {
                    maxVal = tileMax;
                }

                inQueue.FreeTensor(xIn);
            }

            // Pass 2: compute exp(x - max), accumulate sum, write exp to y as workspace
            float sumVal = 0.0f;
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t tileStart = t * TILE_SIZE;
                uint32_t tileLen = (tileStart + TILE_SIZE <= cols) ? TILE_SIZE : (cols - tileStart);

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[row * cols + tileStart], tileLen);
                inQueue.EnQue(xLocal);
                AscendC::LocalTensor<float> xIn = inQueue.DeQue<float>();

                AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
                AscendC::Adds<float>(yLocal, xIn, -maxVal, tileLen);
                AscendC::Exp<float>(yLocal, yLocal, tileLen);

                AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
                AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
                AscendC::ReduceSum<float, true>(scalarLocal, yLocal, reduceTmp, (int32_t)tileLen);
                sumVal += scalarLocal.GetValue(0);

                outQueue.EnQue(yLocal);
                inQueue.FreeTensor(xIn);

                AscendC::LocalTensor<float> yOut = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[row * cols + tileStart], yOut, tileLen);
                outQueue.FreeTensor(yOut);
            }

            AscendC::PipeBarrier<PIPE_ALL>();

            // Pass 3: multiply y by 1/sum to finish softmax
            float invSum = 1.0f / sumVal;
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t tileStart = t * TILE_SIZE;
                uint32_t tileLen = (tileStart + TILE_SIZE <= cols) ? TILE_SIZE : (cols - tileStart);

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(xLocal, yGm[row * cols + tileStart], tileLen);
                inQueue.EnQue(xLocal);
                AscendC::LocalTensor<float> xIn = inQueue.DeQue<float>();

                AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
                AscendC::Muls<float>(yLocal, xIn, invSum, tileLen);

                outQueue.EnQue(yLocal);
                inQueue.FreeTensor(xIn);

                AscendC::LocalTensor<float> yOut = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[row * cols + tileStart], yOut, tileLen);
                outQueue.FreeTensor(yOut);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t cols;
    uint32_t startRow;
    uint32_t endRow;
};

extern "C" __global__ __aicore__ void matmul_dropout_mean_softmax_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulDropoutMeanSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.cols);
    op.Process();
}
