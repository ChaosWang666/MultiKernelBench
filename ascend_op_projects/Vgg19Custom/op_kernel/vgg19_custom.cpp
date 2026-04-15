
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelVgg19 {
public:
    __aicore__ inline KernelVgg19() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalElements, uint32_t batchSize, uint32_t channel, uint32_t height, uint32_t width)
    {
        this->totalElements = totalElements;
        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->batchSize = batchSize;
        this->channel = channel;
        this->height = height;
        this->width = width;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->totalElements / this->blockLength;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->blockLength], this->blockLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simple ReLU operation
        for (uint32_t j = 0; j < this->blockLength; j++) {
            yLocal[j] = (xLocal[j] > 0) ? xLocal[j] : 0;
        }
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->blockLength], yLocal, this->blockLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t totalElements;
    uint32_t blockLength;
    uint32_t batchSize;
    uint32_t channel;
    uint32_t height;
    uint32_t width;
};

extern "C" __global__ __aicore__ void vgg19_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVgg19 op;
    op.Init(x, y, tiling_data.totalElements, tiling_data.batchSize, tiling_data.channel, tiling_data.height, tiling_data.width);
    op.Process();
}
