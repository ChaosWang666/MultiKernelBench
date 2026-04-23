
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulGeluSoftmax {
public:
    __aicore__ inline KernelMatmulGeluSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t cols)
    {
        this->cols = cols;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();

        uint32_t baseRows = totalRows / numBlocks;
        uint32_t extraRows = totalRows % numBlocks;
        uint32_t startRow;
        if (blockIdx < extraRows) {
            this->rowsThisBlock = baseRows + 1;
            startRow = blockIdx * this->rowsThisBlock;
        } else {
            this->rowsThisBlock = baseRows;
            startRow = blockIdx * baseRows + extraRows;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x + startRow * cols, this->rowsThisBlock * cols);
        yGm.SetGlobalBuffer((__gm__ float *)y + startRow * cols, this->rowsThisBlock * cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(tempBuf, cols * sizeof(float));
        pipe.InitBuffer(reduceBuf, 8 * 1024);
        pipe.InitBuffer(scalarBuf, 32);
    }

    __aicore__ inline void Process()
    {
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
        AscendC::LocalTensor<float> tempLocal = tempBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        AscendC::Mul<float>(tempLocal, xLocal, xLocal, cols);
        AscendC::Muls<float>(tempLocal, tempLocal, 0.044715f, cols);
        AscendC::Adds<float>(tempLocal, tempLocal, 1.0f, cols);
        AscendC::Mul<float>(tempLocal, tempLocal, xLocal, cols);
        AscendC::Muls<float>(tempLocal, tempLocal, 1.5957691216f, cols);
        AscendC::Exp<float>(tempLocal, tempLocal, cols);
        AscendC::Adds<float>(tempLocal, tempLocal, 1.0f, cols);
        AscendC::Div<float>(yLocal, xLocal, tempLocal, cols);
        AscendC::Sub<float>(yLocal, xLocal, yLocal, cols);

        AscendC::ReduceMax<float>(scalarLocal, yLocal, reduceTmp, static_cast<int32_t>(cols), false);
        float maxVal = scalarLocal.GetValue(0);
        AscendC::Adds<float>(yLocal, yLocal, -maxVal, cols);
        AscendC::Exp<float>(yLocal, yLocal, cols);
        AscendC::ReduceSum<float, true>(scalarLocal, yLocal, reduceTmp, static_cast<int32_t>(cols));
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t cols;
    uint32_t rowsThisBlock;
};

extern "C" __global__ __aicore__ void matmul_gelu_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulGeluSoftmax op;
    op.Init(x, y, tiling_data.rows, tiling_data.cols);
    op.Process();
}
