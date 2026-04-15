
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTranspose3dClampMinDivide {
public:
    __aicore__ inline KernelConvTranspose3dClampMinDivide() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding, uint32_t depth, uint32_t height, uint32_t width,
                                float minValue, float divisor)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->depth = depth;
        this->.height = height;
        this->width = width;
        this->minValue = minValue;
        this->divisor = divisor;

        // Calculate output dimensions
        this->outDepth = (this->depth - 1) * this->stride - 2 * this->padding + this->kernelSize;
        this->outHeight = (this->height - 1) * this->stride - 2 * this->padding + this->kernelSize;
        this->outWidth = (this->width - 1) * this->stride - 2 * this->padding + this->kernelSize;

        uint32_t totalElements = this->batchSize * this->outChannels * this->outDepth * this->outHeight * this->outWidth;
        this->blockLength = totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        // For simplicity, we assume one tile per block
        CopyIn(0);
        Compute(0);
        CopyOut(0);
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

        // Perform clamp and divide operations
        for (uint32_t i = 0; i < this->blockLength; ++i) {
            float val = xLocal[i];
            if (val < this->minValue) {
                val = this->minValue;
            }
            zLocal[i] = val / this->divisor;
        }

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
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t outDepth;
    uint32_t outHeight;
    uint32_t outWidth;
    float minValue;
    float divisor;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_clamp_min_divide_custom(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dClampMinDivide op;
    op.Init(x, z,
            tiling_data.batchSize,
            tiling_data.inChannels,
            tiling_data.outChannels,
            tiling_data.kernelSize,
            tiling_data.stride,
            tiling_data.padding,
            tiling_data.depth,
            tiling_data.height,
            tiling_data.width,
            tiling_data.minValue,
            tiling_data.divisor);
    op.Process();
}
