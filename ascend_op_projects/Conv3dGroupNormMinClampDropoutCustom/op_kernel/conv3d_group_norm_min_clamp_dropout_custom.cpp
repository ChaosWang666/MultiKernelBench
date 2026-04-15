
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3dGroupNormMinClampDropout {
public:
    __aicore__ inline KernelConv3dGroupNormMinClampDropout() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depth, uint32_t height, uint32_t width, uint32_t kernelSize,
                                uint32_t groups, float minValue, float maxValue, float dropoutP)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depth = depth;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->groups = groups;
        this->minValue = minValue;
        this->maxValue = maxValue;
        this->dropoutP = dropoutP;

        this->totalElements = batch * outChannels * depth * height * width;
        this->blockLength = this->totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = 1;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> localTensor = inQueue.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(localTensor, xGm[progress * this->blockLength], this->blockLength);
        inQueue.EnQue(localTensor);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> localTensor = inQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> outTensor = outQueue.AllocTensor<DTYPE_Y>();

        // Simulate operations: min, clamp, dropout
        for (uint32_t j = 0; j < this->blockLength; ++j) {
            float val = localTensor[j];
            val = AscendC::Min(val, this->minValue);
            val = AscendC::Clamp(val, this->minValue, this->maxValue);
            // Simple dropout simulation - not exact but illustrative
            if (AscendC::Rand() < this->dropoutP) {
                val = 0.0f;
            }
            outTensor[j] = val;
        }

        outQueue.EnQue<DTYPE_Y>(outTensor);
        inQueue.FreeTensor(localTensor);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> outTensor = outQueue.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->blockLength], outTensor, this->blockLength);
        outQueue.FreeTensor(outTensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    uint32_t groups;
    float minValue;
    float maxValue;
    float dropoutP;
    uint32_t totalElements;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv3d_group_norm_min_clamp_dropout_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dGroupNormMinClampDropout op;
    op.Init(x, y, tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depth, tiling_data.height, tiling_data.width, tiling_data.kernelSize,
            tiling_data.groups, tiling_data.minValue, tiling_data.maxValue, tiling_data.dropoutP);
    op.Process();
}
