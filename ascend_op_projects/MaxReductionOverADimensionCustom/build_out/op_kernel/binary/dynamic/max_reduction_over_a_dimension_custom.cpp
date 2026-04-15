
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMaxReduction {
public:
    __aicore__ inline KernelMaxReduction() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dimBefore, uint32_t dimSize, uint32_t dimAfter, uint32_t tileNum)
    {
        this->dimBefore = dimBefore;
        this->dimSize = dimSize;
        this->dimAfter = dimAfter;
        this->totalLength = totalLength; // dimBefore * dimAfter
        
        // Each block handles a portion of the output elements
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->blockLength = (totalLength + blockNum - 1) / blockNum;
        this->startIdx = blockIdx * this->blockLength;
        if (this->startIdx + this->blockLength > totalLength) {
            this->blockLength = totalLength - this->startIdx;
        }
        if (this->startIdx >= totalLength) {
            this->blockLength = 0;
        }
        
        this->tileNum = tileNum;
        
        uint32_t inputSize = dimBefore * dimSize * dimAfter;
        xGm.SetGlobalBuffer((__gm__ float *)x, inputSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        
        // Allocate buffer for one slice along the reduction dimension
        // We need to handle dimSize elements at a time
        // Align to 32 bytes (8 floats)
        uint32_t alignedDimSize = ((dimSize + 7) / 8) * 8;
        if (alignedDimSize < 8) alignedDimSize = 8;
        
        this->alignedDimSize = alignedDimSize;
        
        pipe.InitBuffer(inQueueX, 1, alignedDimSize * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, 8 * sizeof(float)); // at least 32 bytes
    }
    
    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;
        
        for (uint32_t i = 0; i < this->blockLength; i++) {
            uint32_t outIdx = this->startIdx + i;
            // outIdx = beforeIdx * dimAfter + afterIdx
            uint32_t beforeIdx = outIdx / this->dimAfter;
            uint32_t afterIdx = outIdx % this->dimAfter;
            
            if (this->dimAfter == 1) {
                // Reduction along last or contiguous dimension - data is contiguous
                // Input offset: beforeIdx * dimSize * dimAfter + 0 * dimAfter + afterIdx
                //             = beforeIdx * dimSize
                uint32_t srcOffset = beforeIdx * this->dimSize;
                
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[srcOffset], this->alignedDimSize);
                inQueueX.EnQue(xLocal);
                
                xLocal = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                
                AscendC::ReduceMax(yLocal, xLocal, this->dimSize, false);
                
                outQueueY.EnQue(yLocal);
                inQueueX.FreeTensor(xLocal);
                
                yLocal = outQueueY.DeQue<float>();
                // Copy single value
                float maxVal = yLocal.GetValue(0);
                yGm.SetValue(outIdx, maxVal);
                outQueueY.FreeTensor(yLocal);
            } else {
                // General case: gather elements along reduction dim
                // Elements are at: beforeIdx * dimSize * dimAfter + k * dimAfter + afterIdx, k=0..dimSize-1
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                
                // Gather manually
                uint32_t baseOffset = beforeIdx * this->dimSize * this->dimAfter + afterIdx;
                for (uint32_t k = 0; k < this->dimSize; k++) {
                    float val = xGm.GetValue(baseOffset + k * this->dimAfter);
                    xLocal.SetValue(k, val);
                }
                // Pad remaining with very small values
                for (uint32_t k = this->dimSize; k < this->alignedDimSize; k++) {
                    xLocal.SetValue(k, -3.4028235e+38f);
                }
                
                inQueueX.EnQue(xLocal);
                xLocal = inQueueX.DeQue<float>();
                
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                AscendC::ReduceMax(yLocal, xLocal, this->alignedDimSize, false);
                outQueueY.EnQue(yLocal);
                inQueueX.FreeTensor(xLocal);
                
                yLocal = outQueueY.DeQue<float>();
                float maxVal = yLocal.GetValue(0);
                yGm.SetValue(outIdx, maxVal);
                outQueueY.FreeTensor(yLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t dimBefore;
    uint32_t dimSize;
    uint32_t dimAfter;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t startIdx;
    uint32_t tileNum;
    uint32_t alignedDimSize;
};

extern "C" __global__ __aicore__ void max_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxReduction op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dimBefore, tiling_data.dimSize, tiling_data.dimAfter, tiling_data.tileNum);
    op.Process();
}
