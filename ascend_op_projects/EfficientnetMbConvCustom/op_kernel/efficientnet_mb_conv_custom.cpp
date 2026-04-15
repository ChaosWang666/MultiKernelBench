
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelEfficientnetMbConv {
public:
    __aicore__ inline KernelEfficientnetMbConv() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expandWeight, GM_ADDR depthwiseWeight, GM_ADDR projectWeight,
                                GM_ADDR expandBnWeight, GM_ADDR expandBnBias, GM_ADDR depthwiseBnWeight, GM_ADDR depthwiseBnBias,
                                GM_ADDR projectBnWeight, GM_ADDR projectBnBias, GM_ADDR y,
                                uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t kernelSize, uint32_t stride, uint32_t expandRatio, uint32_t useResidual)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->expandRatio = expandRatio;
        this->useResidual = useResidual;
        
        this->hiddenDim = inChannels * expandRatio;
        this->pad = (kernelSize - 1) / 2;
        
        this->xGm.SetGlobalBuffer((__gm__ float *)x, batch * inChannels * height * width * sizeof(float));
        this->expandWeightGm.SetGlobalBuffer((__gm__ float *)expandWeight, inChannels * hiddenDim * 1 * 1 * sizeof(float));
        this->depthwiseWeightGm.SetGlobalBuffer((__gm__ float *)depthwiseWeight, hiddenDim * 1 * kernelSize * kernelSize * sizeof(float));
        this->projectWeightGm.SetGlobalBuffer((__gm__ float *)projectWeight, hiddenDim * outChannels * 1 * 1 * sizeof(float));
        this->expandBnWeightGm.SetGlobalBuffer((__gm__ float *)expandBnWeight, hiddenDim * sizeof(float));
        this->expandBnBiasGm.SetGlobalBuffer((__gm__ float *)expandBnBias, hiddenDim * sizeof(float));
        this->depthwiseBnWeightGm.SetGlobalBuffer((__gm__ float *)depthwiseBnWeight, hiddenDim * sizeof(float));
        this->depthwiseBnBiasGm.SetGlobalBuffer((__gm__ float *)depthwiseBnBias, hiddenDim * sizeof(float));
        this->projectBnWeightGm.SetGlobalBuffer((__gm__ float *)projectBnWeight, outChannels * sizeof(float));
        this->projectBnBiasGm.SetGlobalBuffer((__gm__ float *)projectBnBias, outChannels * sizeof(float));
        this->yGm.SetGlobalBuffer((__gm__ float *)y, batch * outChannels * height * width * sizeof(float));
        
        this->pipe.InitBuffer(inQueueX, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueExpandWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueDepthwiseWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueProjectWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueExpandBnWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueExpandBnBias, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueDepthwiseBnWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueDepthwiseBnBias, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueProjectBnWeight, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(inQueueProjectBnBias, BUFFER_NUM, 1024);
        this->pipe.InitBuffer(outQueueY, BUFFER_NUM, 1024);
    }
    
    __aicore__ inline void Process()
    {
        // Expand conv
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> expandWeightLocal = inQueueExpandWeight.DeQue<float>();
        AscendC::LocalTensor<float> expandBnWeightLocal = inQueueExpandBnWeight.DeQue<float>();
        AscendC::LocalTensor<float> expandBnBiasLocal = inQueueExpandBnBias.DeQue<float>();
        AscendC::LocalTensor<float> expandedLocal = outQueueY.AllocTensor<float>();
        AscendC::Conv2d(expandedLocal, xLocal, expandWeightLocal, expandBnWeightLocal, expandBnBiasLocal, 
                        inChannels, hiddenDim, height, width, 1, 1, 0, 0, 1, 1, 1, 1);
        outQueueY.EnQue<float>(expandedLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueExpandWeight.FreeTensor(expandWeightLocal);
        inQueueExpandBnWeight.FreeTensor(expandBnWeightLocal);
        inQueueExpandBnBias.FreeTensor(expandBnBiasLocal);
        
        // Depthwise conv
        AscendC::LocalTensor<float> expandedLocal2 = outQueueY.DeQue<float>();
        AscendC::LocalTensor<float> depthwiseWeightLocal = inQueueDepthwiseWeight.DeQue<float>();
        AscendC::LocalTensor<float> depthwiseBnWeightLocal = inQueueDepthwiseBnWeight.DeQue<float>();
        AscendC::LocalTensor<float> depthwiseBnBiasLocal = inQueueDepthwiseBnBias.DeQue<float>();
        AscendC::LocalTensor<float> depthwiseLocal = outQueueY.AllocTensor<float>();
        AscendC::Conv2d(depthwiseLocal, expandedLocal2, depthwiseWeightLocal, depthwiseBnWeightLocal, depthwiseBnBiasLocal,
                        hiddenDim, hiddenDim, height, width, kernelSize, kernelSize, pad, pad, stride, stride, 1, hiddenDim);
        outQueueY.EnQue<float>(depthwiseLocal);
        outQueueY.FreeTensor(expandedLocal2);
        inQueueDepthwiseWeight.FreeTensor(depthwiseWeightLocal);
        inQueueDepthwiseBnWeight.FreeTensor(depthwiseBnWeightLocal);
        inQueueDepthwiseBnBias.FreeTensor(depthwiseBnBiasLocal);
        
        // Project conv
        AscendC::LocalTensor<float> depthwiseLocal2 = outQueueY.DeQue<float>();
        AscendC::LocalTensor<float> projectWeightLocal = inQueueProjectWeight.DeQue<float>();
        AscendC::LocalTensor<float> projectBnWeightLocal = inQueueProjectBnWeight.DeQue<float>();
        AscendC::LocalTensor<float> projectBnBiasLocal = inQueueProjectBnBias.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Conv2d(yLocal, depthwiseLocal2, projectWeightLocal, projectBnWeightLocal, projectBnBiasLocal,
                        hiddenDim, outChannels, height, width, 1, 1, 0, 0, 1, 1, 1, 1);
        outQueueY.EnQue<float>(yLocal);
        outQueueY.FreeTensor(depthwiseLocal2);
        inQueueProjectWeight.FreeTensor(projectWeightLocal);
        inQueueProjectBnWeight.FreeTensor(projectBnWeightLocal);
        inQueueProjectBnBias.FreeTensor(projectBnBiasLocal);
        
        // Residual connection
        if (useResidual) {
            AscendC::LocalTensor<float> yLocal2 = outQueueY.DeQue<float>();
            AscendC::LocalTensor<float> xLocal2 = inQueueX.DeQue<float>();
            AscendC::Add(yLocal2, yLocal2, xLocal2, batch * inChannels * height * width);
            outQueueY.EnQue<float>(yLocal2);
            outQueueY.FreeTensor(yLocal2);
            inQueueX.FreeTensor(xLocal2);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueExpandWeight, inQueueDepthwiseWeight, inQueueProjectWeight,
        inQueueExpandBnWeight, inQueueExpandBnBias, inQueueDepthwiseBnWeight, inQueueDepthwiseBnBias, inQueueProjectBnWeight, inQueueProjectBnBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm, expandWeightGm, depthwiseWeightGm, projectWeightGm,
        expandBnWeightGm, expandBnBiasGm, depthwiseBnWeightGm, depthwiseBnBiasGm, projectBnWeightGm, projectBnBiasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch, inChannels, outChannels, height, width, kernelSize, stride, expandRatio, useResidual;
    uint32_t hiddenDim, pad;
};

extern "C" __global__ __aicore__ void efficientnet_mb_conv_custom(
    GM_ADDR x, GM_ADDR expandWeight, GM_ADDR depthwiseWeight, GM_ADDR projectWeight,
    GM_ADDR expandBnWeight, GM_ADDR expandBnBias, GM_ADDR depthwiseBnWeight, GM_ADDR depthwiseBnBias,
    GM_ADDR projectBnWeight, GM_ADDR projectBnBias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelEfficientnetMbConv op;
    op.Init(x, expandWeight, depthwiseWeight, projectWeight, expandBnWeight, expandBnBias, 
            depthwiseBnWeight, depthwiseBnBias, projectBnWeight, projectBnBias, y,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.kernelSize, tiling_data.stride,
            tiling_data.expandRatio, tiling_data.useResidual);
    op.Process();
}
