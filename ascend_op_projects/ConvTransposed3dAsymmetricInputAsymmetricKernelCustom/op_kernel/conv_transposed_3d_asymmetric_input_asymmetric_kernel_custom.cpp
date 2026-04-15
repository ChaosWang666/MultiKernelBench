
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConvTransposed3d {
public:
    __aicore__ inline KernelConvTransposed3d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y, 
                                uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depthIn, uint32_t heightIn, uint32_t widthIn,
                                uint32_t depthOut, uint32_t heightOut, uint32_t widthOut,
                                uint32_t kernelDepth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideDepth, uint32_t strideHeight, uint32_t strideWidth,
                                uint32_t padDepth, uint32_t padHeight, uint32_t padWidth,
                                uint32_t outPadDepth, uint32_t outPadHeight, uint32_t outPadWidth,
                                uint32_t groups)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depthIn = depthIn;
        this->heightIn = heightIn;
        this->widthIn = widthIn;
        this->depthOut = depthOut;
        this->heightOut = heightOut;
        this->widthOut = widthOut;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->strideDepth = strideDepth;
        this->strideHeight = strideHeight;
        this->strideWidth = strideWidth;
        this->padDepth = padDepth;
        this->padHeight = padHeight;
        this->padWidth = padWidth;
        this->outPadDepth = outPadDepth;
        this->outPadHeight = outPadHeight;
        this->outPadWidth = outPadWidth;
        this->groups = groups;

        this->blockLength = batch * outChannels * depthOut * heightOut * widthOut / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batch * inChannels * depthIn * heightIn * widthIn);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, outChannels * inChannels / groups * kernelDepth * kernelHeight * kernelWidth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, batch * outChannels * depthOut * heightOut * widthOut);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->batch * this->outChannels * this->depthOut * this->heightOut * this->widthOut / this->tileLength;
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Placeholder for actual convolution computation
        AscendC::DataCopy(yLocal, xLocal, this->tileLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileLength;
    
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depthIn;
    uint32_t heightIn;
    uint32_t widthIn;
    uint32_t depthOut;
    uint32_t heightOut;
    uint32_t widthOut;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t strideDepth;
    uint32_t strideHeight;
    uint32_t strideWidth;
    uint32_t padDepth;
    uint32_t padHeight;
    uint32_t padWidth;
    uint32_t outPadDepth;
    uint32_t outPadHeight;
    uint32_t outPadWidth;
    uint32_t groups;
};

extern "C" __global__ __aicore__ void conv_transposed_3d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTransposed3d op;
    op.Init(x, weight, y,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depthIn, tiling_data.heightIn, tiling_data.widthIn,
            tiling_data.depthOut, tiling_data.heightOut, tiling_data.widthOut,
            tiling_data.kernelDepth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.strideDepth, tiling_data.strideHeight, tiling_data.strideWidth,
            tiling_data.padDepth, tiling_data.padHeight, tiling_data.padWidth,
            tiling_data.outPadDepth, tiling_data.outPadHeight, tiling_data.outPadWidth,
            tiling_data.groups);
    op.Process();
}
