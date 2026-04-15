
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTranspose3dBatchNormAvgPoolAvgPool {
public:
    __aicore__ inline KernelConvTranspose3dBatchNormAvgPoolAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depth, uint32_t height, uint32_t width, uint32_t kernelSize, uint32_t stride, uint32_t padding)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depth = depth;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;

        this->totalElements = batchSize * outChannels * depth * height * width;
        this->blockLength = this->totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->blockLength], this->blockLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        // Placeholder for actual computation logic
        AscendC::Copy(zLocal, xLocal, this->blockLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->blockLength], zLocal, this->blockLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t totalElements;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_batch_norm_avg_pool_avg_pool_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dBatchNormAvgPoolAvgPool op;
    op.Init(x, z, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depth, tiling_data.height, tiling_data.width, tiling_data.kernelSize,
            tiling_data.stride, tiling_data.padding);
    op.Process();
}
