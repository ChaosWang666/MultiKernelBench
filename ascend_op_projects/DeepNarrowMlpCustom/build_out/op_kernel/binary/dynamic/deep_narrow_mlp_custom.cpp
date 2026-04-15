
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelDeepNarrowMlp {
public:
    __aicore__ inline KernelDeepNarrowMlp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t inputSize, uint32_t outputSize, uint32_t batchSize, uint32_t hiddenLayerSizes)
    {
        this->inputSize = inputSize;
        this->outputSize = outputSize;
        this->batchSize = batchSize;
        this->hiddenLayerSizes = hiddenLayerSizes;
        this->blockLength = inputSize / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
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
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simulate linear layer computation
        AscendC::Mul(yLocal, xLocal, 1.0f, this->tileLength); // Placeholder for actual linear operation
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t inputSize;
    uint32_t outputSize;
    uint32_t batchSize;
    uint32_t hiddenLayerSizes;
};

extern "C" __global__ __aicore__ void deep_narrow_mlp_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeepNarrowMlp op;
    op.Init(x, y, tiling_data.inputSize, tiling_data.outputSize, tiling_data.batchSize, tiling_data.hiddenLayerSizes);
    op.Process();
}
