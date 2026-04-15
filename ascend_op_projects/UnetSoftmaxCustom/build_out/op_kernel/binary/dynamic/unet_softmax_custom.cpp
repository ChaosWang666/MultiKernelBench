
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelUnetSoftmax {
public:
    __aicore__ inline KernelUnetSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t blockSize)
    {
        this->totalLength = totalLength;
        this->blockSize = blockSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = (this->totalLength + this->blockSize - 1) / this->blockSize;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        uint32_t copyLength = (progress + 1) * this->blockSize > this->totalLength ? 
                              this->totalLength - progress * this->blockSize : this->blockSize;
        AscendC::DataCopy(xLocal, xGm[progress * this->blockSize], copyLength);
        inQueue.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        uint32_t computeLength = (progress + 1) * this->blockSize > this->totalLength ? 
                                 this->totalLength - progress * this->blockSize : this->blockSize;
        AscendC::Softmax(yLocal, xLocal, computeLength);
        outQueue.EnQue<float>(yLocal);
        inQueue.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        uint32_t copyLength = (progress + 1) * this->blockSize > this->totalLength ? 
                              this->totalLength - progress * this->blockSize : this->blockSize;
        AscendC::DataCopy(yGm[progress * this->blockSize], yLocal, copyLength);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalLength;
    uint32_t blockSize;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void unet_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelUnetSoftmax op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.blockSize);
    op.Process();
}
