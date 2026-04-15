
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelArgmax {
public:
    __aicore__ inline KernelArgmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dimSize, 
                                 uint32_t outerSize, uint32_t innerSize, uint32_t tileNum)
    {
        this->outerSize = outerSize;
        this->dimSize = dimSize;
        this->innerSize = innerSize;
        this->totalLength = totalLength; // outerSize * innerSize = number of output elements
        
        // Each block processes a portion of the output elements
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->blockLength = (this->totalLength + blockNum - 1) / blockNum;
        uint32_t startIdx = blockIdx * this->blockLength;
        if (startIdx >= this->totalLength) {
            this->blockLength = 0;
            return;
        }
        if (startIdx + this->blockLength > this->totalLength) {
            this->blockLength = this->totalLength - startIdx;
        }
        
        this->startOutputIdx = startIdx;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, outerSize * dimSize * innerSize);
        yGm.SetGlobalBuffer((__gm__ int64_t *)y, totalLength);
    }
    
    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;
        
        // Process each output element assigned to this block
        for (uint32_t i = 0; i < this->blockLength; i++) {
            uint32_t outIdx = this->startOutputIdx + i;
            uint32_t outerIdx = outIdx / this->innerSize;
            uint32_t innerIdx = outIdx % this->innerSize;
            
            // Find argmax along the dim dimension
            // Input index: outerIdx * dimSize * innerSize + d * innerSize + innerIdx
            uint32_t baseOffset = outerIdx * this->dimSize * this->innerSize + innerIdx;
            
            float maxVal = xGm.GetValue(baseOffset);
            int64_t maxIdx = 0;
            
            for (uint32_t d = 1; d < this->dimSize; d++) {
                float val = xGm.GetValue(baseOffset + d * this->innerSize);
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = (int64_t)d;
                }
            }
            
            yGm.SetValue(outIdx, maxIdx);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int64_t> yGm;
    uint32_t outerSize;
    uint32_t dimSize;
    uint32_t innerSize;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t startOutputIdx;
};

extern "C" __global__ __aicore__ void argmax_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelArgmax op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dimSize, 
            tiling_data.outerSize, tiling_data.innerSize, tiling_data.tileNum);
    op.Process();
}
