
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelDensenet121DenseBlock {
public:
    __aicore__ inline KernelDensenet121DenseBlock() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t numLayers, uint32_t numInputFeatures, uint32_t growthRate, uint32_t batchSize, uint32_t height, uint32_t width, uint32_t totalElements)
    {
        this->numLayers = numLayers;
        this->numInputFeatures = numInputFeatures;
        this->growthRate = growthRate;
        this->batchSize = batchSize;
        this->height = height;
        this->width = width;
        this->totalElements = totalElements;
        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = BUFFER_NUM;
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
    uint32_t numLayers;
    uint32_t numInputFeatures;
    uint32_t growthRate;
    uint32_t batchSize;
    uint32_t height;
    uint32_t width;
    uint32_t totalElements;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void densenet121_dense_block_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelDensenet121DenseBlock op;
    op.Init(x, z, tiling_data.numLayers, tiling_data.numInputFeatures, tiling_data.growthRate, tiling_data.batchSize, tiling_data.height, tiling_data.width, tiling_data.totalElements);
    op.Process();
}
