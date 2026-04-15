
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTranspose3dSumResidualAddMultiplyResidualAdd {
public:
    __aicore__ inline KernelConvTranspose3dSumResidualAddMultiplyResidualAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, uint32_t totalLength,
                                uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depth, uint32_t height, uint32_t width,
                                uint32_t kernelDepth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideDepth, uint32_t strideHeight, uint32_t strideWidth,
                                uint32_t padDepth, uint32_t padHeight, uint32_t padWidth,
                                uint32_t outputPadDepth, uint32_t outputPadHeight, uint32_t outputPadWidth)
    {
        this->totalLength = totalLength;
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depth = depth;
        this->height = height;
        this->width = width;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->strideDepth = strideDepth;
        this->strideHeight = strideHeight;
        this->strideWidth = strideWidth;
        this->padDepth = padDepth;
        this->padHeight = padHeight;
        this->padWidth = padWidth;
        this->outputPadDepth = outputPadDepth;
        this->outputPadHeight = outputPadHeight;
        this->outputPadWidth = outputPadWidth;

        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, kernelDepth * kernelHeight * kernelWidth * inChannels * outChannels);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outChannels);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[0], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

        // Simulate ConvTranspose3d operation
        AscendC::Add(zLocal, xLocal, biasLocal, this->tileLength);
        AscendC::Add(zLocal, zLocal, xLocal, this->tileLength);
        AscendC::Mul(zLocal, zLocal, xLocal, this->tileLength);
        AscendC::Add(zLocal, zLocal, xLocal, this->tileLength);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t totalLength;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t strideDepth;
    uint32_t strideHeight;
    uint32_t strideWidth;
    uint32_t padDepth;
    uint32_t padHeight;
    uint32_t padWidth;
    uint32_t outputPadDepth;
    uint32_t outputPadHeight;
    uint32_t outputPadWidth;
};

extern "C" __global__ __aicore__ void conv_transpose3d_sum_residual_add_multiply_residual_add_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dSumResidualAddMultiplyResidualAdd op;
    op.Init(x, weight, bias, z, tiling_data.totalLength,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depth, tiling_data.height, tiling_data.width,
            tiling_data.kernelDepth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.strideDepth, tiling_data.strideHeight, tiling_data.strideWidth,
            tiling_data.padDepth, tiling_data.padHeight, tiling_data.padWidth,
            tiling_data.outputPadDepth, tiling_data.outputPadHeight, tiling_data.outputPadWidth);
    op.Process();
}
