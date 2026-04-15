
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGooglenetInceptionModule {
public:
    __aicore__ inline KernelGooglenetInceptionModule() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t inChannels, uint32_t out1x1, uint32_t reduce3x3, uint32_t out3x3, uint32_t reduce5x5, uint32_t out5x5, uint32_t poolProj, uint32_t batchSize, uint32_t height, uint32_t width, uint32_t totalLength)
    {
        this->inChannels = inChannels;
        this->out1x1 = out1x1;
        this->reduce3x3 = reduce3x3;
        this->out3x3 = out3x3;
        this->reduce5x5 = reduce5x5;
        this->out5x5 = out5x5;
        this->poolProj = poolProj;
        this->batchSize = batchSize;
        this->height = height;
        this->width = width;
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        // Placeholder for actual computation logic
        AscendC::Copy(zLocal, xLocal, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t inChannels;
    uint32_t out1x1;
    uint32_t reduce3x3;
    uint32_t out3x3;
    uint32_t reduce5x5;
    uint32_t out5x5;
    uint32_t poolProj;
    uint32_t batchSize;
    uint32_t height;
    uint32_t width;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void googlenet_inception_module_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGooglenetInceptionModule op;
    op.Init(x, z, tiling_data.inChannels, tiling_data.out1x1, tiling_data.reduce3x3, tiling_data.out3x3, tiling_data.reduce5x5, tiling_data.out5x5, tiling_data.poolProj, tiling_data.batchSize, tiling_data.height, tiling_data.width, tiling_data.totalLength);
    op.Process();
}
