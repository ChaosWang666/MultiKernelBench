
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMeanReduce {
public:
    __aicore__ inline KernelMeanReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dimSize, uint32_t outerSize, uint32_t innerSize, uint32_t tileNum)
    {
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;
        
        // totalLength = outerSize * innerSize = number of output elements
        uint32_t totalOutput = totalLength;
        
        // Distribute output elements across blocks
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->outputPerBlock = totalOutput / blockNum;
        uint32_t remainder = totalOutput % blockNum;
        if (blockIdx < remainder) {
            this->outputPerBlock += 1;
            this->outputOffset = this->outputPerBlock * blockIdx;
        } else {
            this->outputOffset = this->outputPerBlock * blockIdx + remainder;
        }
        
        if (this->outputPerBlock == 0) {
            this->processCount = 0;
            return;
        }
        
        this->processCount = this->outputPerBlock;
        
        // We process innerSize elements at a time when innerSize is large enough
        // Otherwise we process one output element at a time
        
        uint32_t inputTotal = outerSize * dimSize * innerSize;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, inputTotal);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalOutput);
        
        // Tile length for accumulation: process innerSize-aligned chunks
        // For the case innerSize is large (e.g., 4096), we can efficiently vector-add
        // For the case innerSize is small, we need different approach
        
        // Align tileLength to 8 (32 bytes / 4 bytes per float)
        if (this->innerSize >= 8) {
            this->useVectorMode = 1;
            // In vector mode, we process one outer index at a time
            // Each outer index produces innerSize output values by averaging dimSize slices
            this->tileLength = (this->innerSize + 7) / 8 * 8; // align to 8
            uint32_t bufSize = this->tileLength * sizeof(float);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize);
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, bufSize);
        } else {
            this->useVectorMode = 0;
            // In scalar-like mode, we gather dimSize values for each output element
            // Align dimSize to 8
            this->tileLength = (this->dimSize + 7) / 8 * 8;
            uint32_t bufSize = this->tileLength * sizeof(float);
            pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize);
            pipe.InitBuffer(outQueueZ, BUFFER_NUM, bufSize);
        }
    }
    
    __aicore__ inline void Process()
    {
        if (this->processCount == 0) return;
        
        if (this->useVectorMode) {
            ProcessVectorMode();
        } else {
            ProcessScalarMode();
        }
    }

private:
    __aicore__ inline void ProcessVectorMode()
    {
        // Each output element index maps to (outerIdx, innerIdx)
        // outputIdx = outerIdx * innerSize + innerIdx
        // We process outputs in chunks of innerSize (one full outer row at a time)
        
        // Determine which outer indices this block handles
        uint32_t startOut = this->outputOffset;
        uint32_t endOut = this->outputOffset + this->processCount;
        
        uint32_t startOuter = startOut / this->innerSize;
        uint32_t endOuter = (endOut - 1) / this->innerSize;
        
        for (uint32_t outer = startOuter; outer <= endOuter; outer++) {
            uint32_t outStart = outer * this->innerSize;
            uint32_t outEnd = outStart + this->innerSize;
            
            // Clip to this block's range
            uint32_t clippedStart = (outStart >= startOut) ? outStart : startOut;
            uint32_t clippedEnd = (outEnd <= endOut) ? outEnd : endOut;
            
            // For simplicity, if we handle the full inner dimension
            if (clippedStart == outStart && clippedEnd == outEnd) {
                // Full row processing
                // Accumulate: sum over dimSize slices
                AscendC::LocalTensor<float> accLocal = outQueueZ.AllocTensor<float>();
                
                // Zero init accumulator
                AscendC::Duplicate(accLocal, 0.0f, this->tileLength);
                
                for (uint32_t d = 0; d < this->dimSize; d++) {
                    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                    uint32_t srcOffset = outer * this->dimSize * this->innerSize + d * this->innerSize;
                    AscendC::DataCopy(xLocal, xGm[srcOffset], this->tileLength);
                    inQueueX.EnQue(xLocal);
                    xLocal = inQueueX.DeQue<float>();
                    AscendC::Add(accLocal, accLocal, xLocal, this->tileLength);
                    inQueueX.FreeTensor(xLocal);
                }
                
                // Divide by dimSize
                float invDim = 1.0f / (float)this->dimSize;
                AscendC::Muls(accLocal, accLocal, invDim, this->tileLength);
                
                outQueueZ.EnQue(accLocal);
                accLocal = outQueueZ.DeQue<float>();
                AscendC::DataCopy(yGm[outStart], accLocal, this->tileLength);
                outQueueZ.FreeTensor(accLocal);
            } else {
                // Partial row - handle element by element
                for (uint32_t idx = clippedStart; idx < clippedEnd; idx++) {
                    uint32_t innerIdx = idx - outStart;
                    
                    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                    // Gather dimSize values
                    float sum = 0.0f;
                    for (uint32_t d = 0; d < this->dimSize; d++) {
                        uint32_t srcOffset = outer * this->dimSize * this->innerSize + d * this->innerSize + innerIdx;
                        // Simple scalar read via DataCopy of aligned block then pick element
                        // For simplicity, just do aligned read
                        uint32_t alignedOffset = srcOffset / 8 * 8;
                        uint32_t elemOffset = srcOffset - alignedOffset;
                        AscendC::DataCopy(xLocal, xGm[alignedOffset], 8);
                        inQueueX.EnQue(xLocal);
                        xLocal = inQueueX.DeQue<float>();
                        sum += xLocal.GetValue(elemOffset);
                    }
                    
                    float mean = sum / (float)this->dimSize;
                    
                    AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
                    outLocal.SetValue(0, mean);
                    // Write single element via aligned write
                    uint32_t dstAligned = idx / 8 * 8;
                    uint32_t dstElemOff = idx - dstAligned;
                    // Read-modify-write
                    AscendC::DataCopy(outLocal, yGm[dstAligned], 8);
                    outQueueZ.EnQue(outLocal);
                    outLocal = outQueueZ.DeQue<float>();
                    outLocal.SetValue(dstElemOff, mean);
                    AscendC::DataCopy(yGm[dstAligned], outLocal, 8);
                    outQueueZ.FreeTensor(outLocal);
                    
                    inQueueX.FreeTensor(xLocal);
                }
            }
        }
    }
    
    __aicore__ inline void ProcessScalarMode()
    {
        for (uint32_t i = 0; i < this->processCount; i++) {
            uint32_t outIdx = this->outputOffset + i;
            uint32_t outerIdx = outIdx / this->innerSize;
            uint32_t innerIdx = outIdx % this->innerSize;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            
            // Gather dimSize values into xLocal
            for (uint32_t d = 0; d < this->dimSize; d++) {
                uint32_t srcOffset = outerIdx * this->dimSize * this->innerSize + d * this->innerSize + innerIdx;
                uint32_t alignedOffset = srcOffset / 8 * 8;
                uint32_t elemOff = srcOffset - alignedOffset;
                AscendC::LocalTensor<float> tmpLocal = outQueueZ.AllocTensor<float>();
                AscendC::DataCopy(tmpLocal, xGm[alignedOffset], 8);
                outQueueZ.EnQue(tmpLocal);
                tmpLocal = outQueueZ.DeQue<float>();
                xLocal.SetValue(d, tmpLocal.GetValue(elemOff));
                outQueueZ.FreeTensor(tmpLocal);
            }
            
            // Sum reduction
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<float>();
            
            float sum = 0.0f;
            for (uint32_t d = 0; d < this->dimSize; d++) {
                sum += xLocal.GetValue(d);
            }
            float mean = sum / (float)this->dimSize;
            inQueueX.FreeTensor(xLocal);
            
            // Write output
            AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
            uint32_t dstAligned = outIdx / 8 * 8;
            uint32_t dstElemOff = outIdx - dstAligned;
            AscendC::DataCopy(outLocal, yGm[dstAligned], 8);
            outQueueZ.EnQue(outLocal);
            outLocal = outQueueZ.DeQue<float>();
            outLocal.SetValue(dstElemOff, mean);
            AscendC::DataCopy(yGm[dstAligned], outLocal, 8);
            outQueueZ.FreeTensor(outLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t outputPerBlock;
    uint32_t outputOffset;
    uint32_t processCount;
    uint32_t tileLength;
    uint32_t useVectorMode;
};

extern "C" __global__ __aicore__ void mean_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMeanReduce op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dimSize, tiling_data.outerSize, tiling_data.innerSize, tiling_data.tileNum);
    op.Process();
}
