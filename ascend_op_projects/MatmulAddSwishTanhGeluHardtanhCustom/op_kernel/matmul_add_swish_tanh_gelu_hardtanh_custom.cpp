
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulAddSwishTanhGeluHardtanh {
public:
    __aicore__ inline KernelMatmulAddSwishTanhGeluHardtanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR addValue, GM_ADDR y,
                                  uint32_t totalRows, uint32_t cols)
    {
        this->cols = cols;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockDim = AscendC::GetBlockNum();

        uint32_t rowsPerBlock = totalRows / blockDim;
        uint32_t remainder = totalRows % blockDim;

        uint32_t startRow;
        if (blockIdx < remainder) {
            this->rowCount = rowsPerBlock + 1;
            startRow = blockIdx * this->rowCount;
        } else {
            this->rowCount = rowsPerBlock;
            startRow = remainder * (rowsPerBlock + 1) + (blockIdx - remainder) * rowsPerBlock;
        }

        if (this->rowCount == 0) {
            xGm.SetGlobalBuffer((__gm__ float*)x, 0);
            yGm.SetGlobalBuffer((__gm__ float*)y, 0);
            addGm.SetGlobalBuffer((__gm__ float*)addValue, cols);
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * cols, this->rowCount * cols);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * cols, this->rowCount * cols);
        addGm.SetGlobalBuffer((__gm__ float*)addValue, cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, cols * sizeof(float));
        pipe.InitBuffer(addQueue, 1, cols * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->rowCount == 0) {
            return;
        }

        AscendC::LocalTensor<float> addTmp = addQueue.AllocTensor<float>();
        AscendC::DataCopy(addTmp, addGm, this->cols);
        addQueue.EnQue(addTmp);
        AscendC::LocalTensor<float> addLocal = addQueue.DeQue<float>();

        for (uint32_t i = 0; i < this->rowCount; i++) {
            CopyIn(i);
            Compute(i, addLocal);
            CopyOut(i);
        }

        addQueue.FreeTensor(addLocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->cols], this->cols);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress, AscendC::LocalTensor<float>& addLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Step 1: x = x + add_value
        AscendC::Add(xLocal, xLocal, addLocal, this->cols);

        // Step 2: Swish: x = x * sigmoid(x)
        AscendC::Sigmoid(yLocal, xLocal, this->cols);
        AscendC::Mul(xLocal, xLocal, yLocal, this->cols);

        // Step 3: Tanh(x)
        AscendC::Tanh(xLocal, xLocal, this->cols);

        // Step 4: GELU(x)
        AscendC::Gelu(xLocal, xLocal, this->cols);

        // Step 5: Hardtanh: clamp to [-1, 1]
        AscendC::Maxs(xLocal, xLocal, (float)-1.0f, this->cols);
        AscendC::Mins(yLocal, xLocal, (float)1.0f, this->cols);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->cols], yLocal, this->cols);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> addQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> addGm;
    uint32_t cols;
    uint32_t rowCount;
};

extern "C" __global__ __aicore__ void matmul_add_swish_tanh_gelu_hardtanh_custom(
    GM_ADDR x, GM_ADDR add_value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulAddSwishTanhGeluHardtanh op;
    op.Init(x, add_value, y, tiling_data.totalRows, tiling_data.cols);
    op.Process();
}
