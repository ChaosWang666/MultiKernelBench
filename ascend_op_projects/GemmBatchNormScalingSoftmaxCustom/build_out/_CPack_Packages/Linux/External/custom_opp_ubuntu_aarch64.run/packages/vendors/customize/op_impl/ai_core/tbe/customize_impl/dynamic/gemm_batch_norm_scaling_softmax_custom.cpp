
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t cols)
    {
        this->cols = cols;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t rowsPerBlockBase = totalRows / blockNum;
        uint32_t extraRows = totalRows % blockNum;

        uint32_t startRow;
        uint32_t rowsThisBlockLocal;
        if (blockIdx < extraRows) {
            rowsThisBlockLocal = rowsPerBlockBase + 1;
            startRow = blockIdx * rowsThisBlockLocal;
        } else {
            rowsThisBlockLocal = rowsPerBlockBase;
            startRow = extraRows * (rowsPerBlockBase + 1) + (blockIdx - extraRows) * rowsPerBlockBase;
        }

        this->rowsThisBlock = rowsThisBlockLocal;

        if (rowsThisBlockLocal == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x + startRow * cols, rowsThisBlockLocal * cols);
        yGm.SetGlobalBuffer((__gm__ float *)y + startRow * cols, rowsThisBlockLocal * cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32 * 1024);
        pipe.InitBuffer(scalarBuf, 64);
    }

    __aicore__ inline void Process()
    {
        if (rowsThisBlock == 0) {
            return;
        }
        for (uint32_t i = 0; i < rowsThisBlock; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * cols], cols);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        AscendC::ReduceMax<float>(scalarLocal, xLocal, reduceTmp, (int32_t)cols, false);
        float maxVal = scalarLocal.GetValue(0);

        AscendC::Adds<float>(yLocal, xLocal, -maxVal, cols);

        AscendC::Exp<float>(yLocal, yLocal, cols);

        AscendC::ReduceSum<float>(scalarLocal, yLocal, reduceTmp, (int32_t)cols);
        float sumVal = scalarLocal.GetValue(0);
        float invSum = 1.0f / sumVal;

        AscendC::Muls<float>(yLocal, yLocal, invSum, cols);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * cols], yLocal, cols);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t rowsThisBlock;
    uint32_t cols;
};

extern "C" __global__ __aicore__ void gemm_batch_norm_scaling_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.cols);
    op.Process();
}
