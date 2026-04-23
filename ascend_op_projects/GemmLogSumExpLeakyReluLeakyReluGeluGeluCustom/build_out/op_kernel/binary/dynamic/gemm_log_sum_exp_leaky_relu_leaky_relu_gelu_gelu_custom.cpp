
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SCALAR_COUNT = 8;

class KernelLseAct {
public:
    __aicore__ inline KernelLseAct() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t cols)
    {
        this->cols = cols;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t baseRows = totalRows / blockNum;
        uint32_t extraRows = totalRows % blockNum;
        uint32_t startRow;
        if (blockIdx < extraRows) {
            this->rowsThisBlock = baseRows + 1;
            startRow = blockIdx * this->rowsThisBlock;
        } else {
            this->rowsThisBlock = baseRows;
            startRow = extraRows * (baseRows + 1) + (blockIdx - extraRows) * baseRows;
        }

        if (this->rowsThisBlock == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * cols, this->rowsThisBlock * cols);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow, this->rowsThisBlock);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
        pipe.InitBuffer(reduceBuf, 32 * 1024);
        pipe.InitBuffer(tmpBuf, 256);
    }

    __aicore__ inline void Process()
    {
        if (this->rowsThisBlock == 0) {
            return;
        }
        for (uint32_t i = 0; i < this->rowsThisBlock; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[i * this->cols], this->cols);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t i)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();

        AscendC::ReduceMax<float>(yLocal, xLocal, reduceTmp, (int32_t)this->cols, false);
        float maxVal = yLocal.GetValue(0);

        AscendC::Adds<float>(xLocal, xLocal, -maxVal, this->cols);
        AscendC::Exp<float>(xLocal, xLocal, this->cols);

        AscendC::ReduceSum<float, true>(yLocal, xLocal, reduceTmp, (int32_t)this->cols);
        inQueueX.FreeTensor(xLocal);

        AscendC::Log<float>(yLocal, yLocal, SCALAR_COUNT);
        AscendC::Adds<float>(yLocal, yLocal, maxVal, SCALAR_COUNT);

        AscendC::LeakyRelu<float>(yLocal, yLocal, (float)0.01, SCALAR_COUNT);
        AscendC::LeakyRelu<float>(yLocal, yLocal, (float)0.01, SCALAR_COUNT);

        ApplyGelu(yLocal, tmp);
        ApplyGelu(yLocal, tmp);

        outQueueY.EnQue<float>(yLocal);
    }

    __aicore__ inline void ApplyGelu(AscendC::LocalTensor<float>& x, AscendC::LocalTensor<float>& tmp)
    {
        AscendC::Mul<float>(tmp, x, x, SCALAR_COUNT);
        AscendC::Mul<float>(tmp, tmp, x, SCALAR_COUNT);
        AscendC::Muls<float>(tmp, tmp, (float)0.044715, SCALAR_COUNT);
        AscendC::Add<float>(tmp, tmp, x, SCALAR_COUNT);
        AscendC::Muls<float>(tmp, tmp, (float)0.7978845608028654, SCALAR_COUNT);
        AscendC::Tanh<float>(tmp, tmp, SCALAR_COUNT);
        AscendC::Adds<float>(tmp, tmp, (float)1.0, SCALAR_COUNT);
        AscendC::Mul<float>(tmp, tmp, x, SCALAR_COUNT);
        AscendC::Muls<float>(x, tmp, (float)0.5, SCALAR_COUNT);
    }

    __aicore__ inline void CopyOut(uint32_t i)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[i], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t cols;
    uint32_t rowsThisBlock;
};

extern "C" __global__ __aicore__ void gemm_log_sum_exp_leaky_relu_leaky_relu_gelu_gelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLseAct op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.cols);
    op.Process();
}
