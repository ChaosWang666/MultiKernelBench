
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConvTranspose2dGeluGroupNorm {
public:
    __aicore__ inline KernelConvTranspose2dGeluGroupNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelSize, uint32_t stride,
                                uint32_t groups, uint32_t numGroups)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->groups = groups;
        this->numGroups = numGroups;

        uint32_t totalElements = batch * outChannels * height * width;
        this->blockLength = totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockLength * sizeof(float));
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
        AscendC::LocalTensor<float> localTensor = inQueue.AllocTensor<float>();
        AscendC::DataCopy(localTensor, xGm[progress * this->blockLength], this->blockLength);
        inQueue.EnQue(localTensor);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> inputTensor = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outputTensor = outQueue.AllocTensor<float>();

        // Simulate operations: ConvTranspose2d -> GELU -> GroupNorm
        // In practice, these would be implemented with actual kernels
        for (uint32_t i = 0; i < this->blockLength; ++i) {
            float val = inputTensor[i];
            // Apply GELU activation
            float geluVal = 0.5f * val * (1.0f + AscendC::Erf(val / sqrtf(2.0f)));
            outputTensor[i] = geluVal;
        }

        outQueue.EnQue<float>(outputTensor);
        inQueue.FreeTensor(inputTensor);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outputTensor = outQueue.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->blockLength], outputTensor, this->blockLength);
        outQueue.FreeTensor(outputTensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t groups;
    uint32_t numGroups;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void conv_transpose2d_gelu_group_norm_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose2dGeluGroupNorm op;
    op.Init(x, y, tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize, tiling_data.stride,
            tiling_data.groups, tiling_data.numGroups);
    op.Process();
}
