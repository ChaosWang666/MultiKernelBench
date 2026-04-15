
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmaxBiasScaleSigmoid {
public:
    __aicore__ inline KernelSoftmaxBiasScaleSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t height, uint32_t width,
                                 float scalingFactor, uint32_t totalLength, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->height = height;
        this->width = width;
        this->scalingFactor = scalingFactor;
        this->hw = height * width;
        
        // Each block processes a portion of spatial locations across all batches
        // Total spatial work items = batchSize * height * width
        uint32_t totalSpatial = batchSize * hw;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->spatialStart = (totalSpatial / numBlocks) * blockIdx;
        uint32_t spatialEnd = (blockIdx == numBlocks - 1) ? totalSpatial : (totalSpatial / numBlocks) * (blockIdx + 1);
        this->spatialCount = spatialEnd - this->spatialStart;
        
        // Align channels to 32 bytes / 4 = 8 floats
        this->channelsAligned = ((channels + 7) / 8) * 8;
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, channels);
        zGm.SetGlobalBuffer((__gm__ float*)z, totalLength);
        
        // We process one spatial location at a time, need channelsAligned buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->channelsAligned * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->channelsAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf1, BUFFER_NUM, this->channelsAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf2, BUFFER_NUM, this->channelsAligned * sizeof(float));
        pipe.InitBuffer(biasBuf, 1, this->channelsAligned * sizeof(float));
        
        // Load bias into local buffer
        AscendC::LocalTensor<float> biasLocal = biasBuf.AllocTensor<float>();
        // bias shape is (channels, 1, 1), stored contiguously
        AscendC::DataCopy(biasLocal, biasGm[0], this->channelsAligned);
        biasBuf.EnQue(biasLocal);
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->spatialCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t GetGlobalOffset(uint32_t spatialIdx)
    {
        // spatialIdx is local index, actual = spatialStart + spatialIdx
        uint32_t actualSpatial = this->spatialStart + spatialIdx;
        uint32_t b = actualSpatial / this->hw;
        uint32_t spatialPos = actualSpatial % this->hw;
        uint32_t h = spatialPos / this->width;
        uint32_t w = spatialPos % this->width;
        // NCHW layout: offset = b*C*H*W + 0*H*W + h*W + w
        // We need to gather C values at stride H*W
        return b * this->channels * this->hw + h * this->width + w;
    }
    
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t baseOffset = GetGlobalOffset(progress);
        
        // Gather channel values: x[b, c, h, w] for all c
        // Stride between channels is hw
        for (uint32_t c = 0; c < this->channels; c++) {
            // Use scalar copy approach - copy one element at a time
            float val = *(((__gm__ float*)xGm.GetPhyAddr()) + baseOffset + c * this->hw);
            xLocal.SetValue(c, val);
        }
        // Pad remaining
        for (uint32_t c = this->channels; c < this->channelsAligned; c++) {
            xLocal.SetValue(c, 0.0f);
        }
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = biasBuf.DeQue<float>();
        
        uint32_t len = this->channelsAligned;
        
        // Step 1: Softmax along channels
        // Find max for numerical stability
        AscendC::ReduceMax(tmp1, xLocal, tmp2, len);
        float maxVal = tmp1.GetValue(0);
        
        // Subtract max: xLocal = xLocal - maxVal
        AscendC::Adds(xLocal, xLocal, -maxVal, len);
        
        // Exp
        AscendC::Exp(xLocal, xLocal, len);
        
        // Zero out padded channels
        for (uint32_t c = this->channels; c < this->channelsAligned; c++) {
            xLocal.SetValue(c, 0.0f);
        }
        
        // Sum of exponentials
        AscendC::ReduceSum(tmp1, xLocal, tmp2, len);
        float sumVal = tmp1.GetValue(0);
        
        // Divide by sum
        if (sumVal > 0.0f) {
            float invSum = 1.0f / sumVal;
            AscendC::Muls(xLocal, xLocal, invSum, len);
        }
        
        // Step 2: Add bias
        AscendC::Add(xLocal, xLocal, biasLocal, len);
        
        // Step 3: Scale
        AscendC::Muls(xLocal, xLocal, this->scalingFactor, len);
        
        // Step 4: Sigmoid: 1/(1+exp(-x))
        // Negate
        AscendC::Muls(zLocal, xLocal, -1.0f, len);
        // Exp
        AscendC::Exp(zLocal, zLocal, len);
        // Add 1
        AscendC::Adds(zLocal, zLocal, 1.0f, len);
        // Reciprocal
        AscendC::Reciprocal(zLocal, zLocal, len);
        
        biasBuf.EnQue(biasLocal);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        tmpBuf1.FreeTensor(tmp1);
        tmpBuf2.FreeTensor(tmp2);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        uint32_t baseOffset = GetGlobalOffset(progress);
        
        // Scatter channel values back
        for (uint32_t c = 0; c < this->channels; c++) {
            float val = zLocal.GetValue(c);
            *(((__gm__ float*)zGm.GetPhyAddr()) + baseOffset + c * this->hw) = val;
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf1, tmpBuf2;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> biasBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint32_t hw;
    uint32_t channelsAligned;
    uint32_t spatialStart;
    uint32_t spatialCount;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void convtranspose2d_softmax_biasadd_scaling_sigmoid_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmaxBiasScaleSigmoid op;
    op.Init(x, bias, z,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.height, tiling_data.width,
            tiling_data.scalingFactor, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
