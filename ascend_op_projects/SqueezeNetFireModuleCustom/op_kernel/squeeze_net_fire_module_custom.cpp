
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelSqueezeNetFireModule {
public:
    __aicore__ inline KernelSqueezeNetFireModule() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t squeezeChannels,
                                uint32_t expand1x1Channels, uint32_t expand3x3Channels)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->squeezeChannels = squeezeChannels;
        this->expand1x1Channels = expand1x1Channels;
        this->expand3x3Channels = expand3x3Channels;
        
        this->totalElements = batchSize * inChannels * height * width;
        this->blockLength = this->totalElements / AscendC::GetBlockNum();
        
        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
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
        AscendC::LocalTensor<float> local = inQueue.AllocTensor<float>();
        AscendC::DataCopy(local, xGm[0], this->blockLength);
        inQueue.EnQue(local);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> local = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        // Placeholder for actual computation logic
        AscendC::Memcpy(outLocal, local, this->blockLength * sizeof(float));
        outQueue.EnQue<float>(outLocal);
        inQueue.FreeTensor(local);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(zGm[0], outLocal, this->blockLength);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t squeezeChannels;
    uint32_t expand1x1Channels;
    uint32_t expand3x3Channels;
    uint32_t totalElements;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void squeeze_net_fire_module_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSqueezeNetFireModule op;
    op.Init(x, z, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.squeezeChannels,
            tiling_data.expand1x1Channels, tiling_data.expand3x3Channels);
    op.Process();
}
