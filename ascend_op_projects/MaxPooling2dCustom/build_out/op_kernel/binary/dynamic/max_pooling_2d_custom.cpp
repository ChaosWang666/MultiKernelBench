
#include "kernel_operator.h"

class KernelMaxPool2d {
public:
    __aicore__ inline KernelMaxPool2d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t height, uint32_t width,
                                 uint32_t kernelSize, uint32_t stride,
                                 uint32_t padding,
                                 uint32_t outHeight, uint32_t outWidth)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->height = height;
        this->width = width;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;
        this->outHeight = outHeight;
        this->outWidth = outWidth;
        
        uint32_t totalPlanes = batchSize * channels;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->planesPerBlock = (totalPlanes + numBlocks - 1) / numBlocks;
        this->startPlane = blockIdx * this->planesPerBlock;
        if (this->startPlane > totalPlanes) this->startPlane = totalPlanes;
        this->endPlane = this->startPlane + this->planesPerBlock;
        if (this->endPlane > totalPlanes) this->endPlane = totalPlanes;
        
        uint32_t inPlaneSize = height * width;
        uint32_t outPlaneSize = outHeight * outWidth;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * channels * inPlaneSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * channels * outPlaneSize);
        
        // Allocate buffer for one input row's worth of data for processing
        // We'll process output row by row
        // We need kernelSize rows of input for one output row
        uint32_t alignedWidth = ((width + 7) / 8) * 8;
        uint32_t rowBufSize = alignedWidth * sizeof(float);
        
        // Allocate buffer for one output row
        uint32_t alignedOutWidth = ((outWidth + 7) / 8) * 8;
        uint32_t outRowBufSize = alignedOutWidth * sizeof(float);
        
        pipe.InitBuffer(inQueue, 1, rowBufSize);
        pipe.InitBuffer(outQueue, 1, outRowBufSize);
        pipe.InitBuffer(tmpBuf, 1, alignedOutWidth * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        uint32_t inPlaneSize = height * width;
        uint32_t outPlaneSize = outHeight * outWidth;
        
        for (uint32_t p = startPlane; p < endPlane; p++) {
            float* inBase = (float*)(xGm.GetPhyAddr()) + p * inPlaneSize;
            float* outBase = (float*)(yGm.GetPhyAddr()) + p * outPlaneSize;
            
            for (uint32_t oh = 0; oh < outHeight; oh++) {
                // For each output row, compute max pooling
                // First, initialize output row to -FLT_MAX
                uint32_t alignedOutWidth = ((outWidth + 7) / 8) * 8;
                
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                
                // Initialize to very negative value
                AscendC::Duplicate(outLocal, (float)(-3.402823e+38f), alignedOutWidth);
                
                for (uint32_t kh = 0; kh < kernelSize; kh++) {
                    int32_t ih = (int32_t)(oh * stride) - (int32_t)padding + (int32_t)kh;
                    if (ih < 0 || ih >= (int32_t)height) continue;
                    
                    for (uint32_t kw = 0; kw < kernelSize; kw++) {
                        // For each output position in this row
                        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
                        
                        // Build the values for this kernel position across all output columns
                        for (uint32_t ow = 0; ow < outWidth; ow++) {
                            int32_t iw = (int32_t)(ow * stride) - (int32_t)padding + (int32_t)kw;
                            float val;
                            if (iw < 0 || iw >= (int32_t)width) {
                                val = -3.402823e+38f;
                            } else {
                                val = *((float*)(xGm.GetPhyAddr()) + p * inPlaneSize + ih * width + iw);
                            }
                            tmpLocal.SetValue(ow, val);
                        }
                        // Pad remaining
                        for (uint32_t ow = outWidth; ow < alignedOutWidth; ow++) {
                            tmpLocal.SetValue(ow, -3.402823e+38f);
                        }
                        
                        // Max with current output
                        AscendC::Max(outLocal, outLocal, tmpLocal, alignedOutWidth);
                    }
                }
                
                // Copy output row to global memory
                // We need to copy only outWidth elements but DataCopy needs aligned
                AscendC::DataCopy(yGm[p * outPlaneSize + oh * outWidth], outLocal, alignedOutWidth <= outWidth ? alignedOutWidth : outWidth);
                
                outQueue.FreeTensor(outLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    
    uint32_t batchSize, channels, height, width;
    uint32_t kernelSize, stride, padding;
    uint32_t outHeight, outWidth;
    uint32_t planesPerBlock, startPlane, endPlane;
};

extern "C" __global__ __aicore__ void max_pooling_2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPool2d op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.height, tiling_data.width,
            tiling_data.kernelSize, tiling_data.stride,
            tiling_data.padding,
            tiling_data.outHeight, tiling_data.outWidth);
    op.Process();
}
