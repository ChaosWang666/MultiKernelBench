
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelShallowWideMlp {
public:
    __aicore__ inline KernelShallowWideMlp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w0, GM_ADDR b0, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR z,
                                uint32_t batchSize, uint32_t inputSize, uint32_t hiddenSize0, uint32_t hiddenSize1, uint32_t outputSize, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize0 = hiddenSize0;
        this->hiddenSize1 = hiddenSize1;
        this->outputSize = outputSize;
        this->tileNum = tileNum;
        this->blockLength = batchSize * inputSize / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        w0Gm.SetGlobalBuffer((__gm__ DTYPE_W0 *)w0, hiddenSize0 * inputSize);
        b0Gm.SetGlobalBuffer((__gm__ DTYPE_B0 *)b0, hiddenSize0);
        w1Gm.SetGlobalBuffer((__gm__ DTYPE_W1 *)w1, hiddenSize1 * hiddenSize0);
        b1Gm.SetGlobalBuffer((__gm__ DTYPE_B1 *)b1, hiddenSize1);
        w2Gm.SetGlobalBuffer((__gm__ DTYPE_W2 *)w2, outputSize * hiddenSize1);
        b2Gm.SetGlobalBuffer((__gm__ DTYPE_B2 *)b2, outputSize);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, batchSize * outputSize);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW0, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W0));
        pipe.InitBuffer(inQueueB0, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B0));
        pipe.InitBuffer(inQueueW1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W1));
        pipe.InitBuffer(inQueueB1, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B1));
        pipe.InitBuffer(inQueueW2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W2));
        pipe.InitBuffer(inQueueB2, BUFFER_NUM, this->tileLength * sizeof(DTYPE_B2));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        // First linear layer
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute1(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute1(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W0> w0Local = inQueueW0.AllocTensor<DTYPE_W0>();
        AscendC::LocalTensor<DTYPE_B0> b0Local = inQueueB0.AllocTensor<DTYPE_B0>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::MatMul(zLocal, xLocal, w0Local, b0Local, this->tileLength, this->inputSize, this->hiddenSize0);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueW0, inQueueB0, inQueueW1, inQueueB1, inQueueW2, inQueueB2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_W0> w0Gm;
    AscendC::GlobalTensor<DTYPE_B0> b0Gm;
    AscendC::GlobalTensor<DTYPE_W1> w1Gm;
    AscendC::GlobalTensor<DTYPE_B1> b1Gm;
    AscendC::GlobalTensor<DTYPE_W2> w2Gm;
    AscendC::GlobalTensor<DTYPE_B2> b2Gm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batchSize;
    uint32_t inputSize;
    uint32_t hiddenSize0;
    uint32_t hiddenSize1;
    uint32_t outputSize;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void shallow_wide_mlp_custom(
    GM_ADDR x, GM_ADDR w0, GM_ADDR b0, GM_ADDR w1, GM_ADDR b1, GM_ADDR w2, GM_ADDR b2, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelShallowWideMlp op;
    op.Init(x, w0, b0, w1, b1, w2, b2, z,
            tiling_data.batchSize, tiling_data.inputSize, tiling_data.hiddenSize0, tiling_data.hiddenSize1, tiling_data.outputSize, tiling_data.tileNum);
    op.Process();
}
