
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelRegnet {
public:
    __aicore__ inline KernelRegnet() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t inputChannels, uint32_t height, uint32_t width, uint32_t outputChannels)
    {
        this->batchSize = batchSize;
        this->inputChannels = inputChannels;
        this->height = height;
        this->width = width;
        this->outputChannels = outputChannels;
        
        this->totalElements = batchSize * inputChannels * height * width;
        this->elementsPerBatch = inputChannels * height * width;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, this->totalElements);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * outputChannels);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->elementsPerBatch * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, batchSize * outputChannels * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; ++batch) {
            CopyIn(batch);
            Compute(batch);
            CopyOut(batch);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[batch * elementsPerBatch], elementsPerBatch);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint32_t batch)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // Simple averaging operation for demonstration
        for (uint32_t i = 0; i < outputChannels; ++i) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < elementsPerBatch; ++j) {
                sum += xLocal[j];
            }
            yLocal[i] = sum / elementsPerBatch;
        }
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t batch)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[batch * outputChannels], yLocal, outputChannels);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inputChannels;
    uint32_t height;
    uint32_t width;
    uint32_t outputChannels;
    uint32_t totalElements;
    uint32_t elementsPerBatch;
};

extern "C" __global__ __aicore__ void regnet_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRegnet op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.inputChannels, tiling_data.height, tiling_data.width, tiling_data.outputChannels);
    op.Process();
}
