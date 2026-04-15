
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelDensenet121TransitionLayer {
public:
    __aicore__ inline KernelDensenet121TransitionLayer() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batch, uint32_t height, uint32_t width,
                                uint32_t numInputFeatures, uint32_t numOutputFeatures)
    {
        this->batch = batch;
        this->height = height;
        this->width = width;
        this->numInputFeatures = numInputFeatures;
        this->numOutputFeatures = numOutputFeatures;
        this->totalElements = batch * height * width * numInputFeatures;
        this->blockLength = totalElements / AscendC::GetBlockNum();

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->blockLength / 1024;
        if (this->blockLength % 1024 != 0) {
            loopCount++;
        }
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
        AscendC::DataCopy(localTensor, xGm[progress * 1024], 1024);
        inQueue.EnQue(localTensor);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> localTensor = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outTensor = outQueue.AllocTensor<float>();
        // Placeholder for actual computation logic
        AscendC::Copy(outTensor, localTensor, 1024);
        outQueue.EnQue<float>(outTensor);
        inQueue.FreeTensor(localTensor);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outTensor = outQueue.DeQue<float>();
        AscendC::DataCopy(yGm[progress * 1024], outTensor, 1024);
        outQueue.FreeTensor(outTensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t numInputFeatures;
    uint32_t numOutputFeatures;
    uint32_t totalElements;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void densenet121_transition_layer_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelDensenet121TransitionLayer op;
    op.Init(x, y, tiling_data.batch, tiling_data.height, tiling_data.width,
            tiling_data.numInputFeatures, tiling_data.numOutputFeatures);
    op.Process();
}
