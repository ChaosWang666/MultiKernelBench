
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3dHardswishReluSoftmaxMean {
public:
    __aicore__ inline KernelConv3dHardswishReluSoftmaxMean() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depth, uint32_t height, uint32_t width,
                                uint32_t kernelDepth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t padD, uint32_t padH, uint32_t padW,
                                uint32_t strideD, uint32_t strideH, uint32_t strideW,
                                uint32_t dilationD, uint32_t dilationH, uint32_t dilationW,
                                uint32_t group, uint32_t totalElements)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depth = depth;
        this->height = height;
        this->width = width;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->padD = padD;
        this->padH = padH;
        this->padW = padW;
        this->strideD = strideD;
        this->strideH = strideH;
        this->strideW = strideW;
        this->dilationD = dilationD;
        this->dilationH = dilationH;
        this->dilationW = dilationW;
        this->group = group;
        this->totalElements = totalElements;

        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, outChannels * inChannels * kernelDepth * kernelHeight * kernelWidth);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> weightLocal = inQueueWeight.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // Simplified computation for demonstration
        AscendC::Add(yLocal, xLocal, weightLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, biasLocal, this->tileLength);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t padD;
    uint32_t padH;
    uint32_t padW;
    uint32_t strideD;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t dilationD;
    uint32_t dilationH;
    uint32_t dilationW;
    uint32_t group;
    uint32_t totalElements;
};

extern "C" __global__ __aicore__ void conv3d_hardswish_relu_softmax_mean_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dHardswishReluSoftmaxMean op;
    op.Init(x, weight, bias, y,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depth, tiling_data.height, tiling_data.width,
            tiling_data.kernelDepth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.padD, tiling_data.padH, tiling_data.padW,
            tiling_data.strideD, tiling_data.strideH, tiling_data.strideW,
            tiling_data.dilationD, tiling_data.dilationH, tiling_data.dilationW,
            tiling_data.group, tiling_data.totalElements);
    op.Process();
}
