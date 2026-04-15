
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelMlp {
public:
    __aicore__ inline KernelMlp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, uint32_t batchSize, uint32_t inputSize, uint32_t outputSize, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->tileNum = tileNum;
        this->blockLength = outputSize / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inputSize);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, inputSize * outputSize);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outputSize);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batchSize * outputSize);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; batch++) {
            ProcessBatch(batch);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchId)
    {
        int32_t loopCount = this->outputSize / this->tileNum / BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, batchId);
            Compute(i, batchId);
            CopyOut(i, batchId);
        }
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t batchId)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::DataCopy(xLocal, xGm[batchId * inputSize], inputSize);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength * inputSize], this->tileLength * inputSize);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t batchId)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::MatMul(yLocal, xLocal, weightLocal, this->tileLength, inputSize, 1.0f, 0.0f);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t batchId)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[batchId * outputSize + progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t tileNum;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void mlp_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMlp op;
    op.Init(x, weight, bias, y, tiling_data.batchSize, tiling_data.inputSize, tiling_data.outputSize, tiling_data.tileNum);
    op.Process();
}
