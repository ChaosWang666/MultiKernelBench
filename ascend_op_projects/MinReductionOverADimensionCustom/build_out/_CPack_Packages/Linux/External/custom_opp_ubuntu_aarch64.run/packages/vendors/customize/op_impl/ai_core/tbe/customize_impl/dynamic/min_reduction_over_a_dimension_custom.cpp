
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMinReduction {
public:
    __aicore__ inline KernelMinReduction() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dim, 
                                 uint32_t dimSize, uint32_t outerSize, uint32_t innerSize, uint32_t tileNum)
    {
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;
        
        // Total number of output elements = outerSize * innerSize
        uint32_t totalOut = outerSize * innerSize;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Each block handles a portion of the output elements
        this->blockOutLen = totalOut / blockNum;
        uint32_t remainder = totalOut % blockNum;
        
        uint32_t startOut;
        if (blockIdx < remainder) {
            this->blockOutLen += 1;
            startOut = blockIdx * this->blockOutLen;
        } else {
            startOut = blockIdx * this->blockOutLen + remainder;
        }
        
        if (this->blockOutLen == 0) {
            this->blockOutLen = 0;
            return;
        }
        
        this->startIdx = startOut;
        
        // Align tile length to 32 bytes (8 floats)
        uint32_t alignNum = 8;
        
        // We process output elements in tiles
        // Each tile processes tileLength output elements at a time
        this->tileNum = tileNum;
        if (this->tileNum > this->blockOutLen) {
            this->tileNum = 1;
        }
        this->tileLength = this->blockOutLen / this->tileNum;
        // Align tileLength up to alignNum
        if (this->tileLength < alignNum) {
            this->tileLength = alignNum;
            this->tileNum = (this->blockOutLen + this->tileLength - 1) / this->tileLength;
        }
        this->lastTileLength = this->blockOutLen - (this->tileNum - 1) * this->tileLength;
        
        // Set up global memory
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + startOut, this->blockOutLen);
        
        // We need buffers for reading one slice at a time and for the running min
        // tileLength floats for input slice, tileLength floats for min accumulator
        uint32_t maxTile = this->tileLength > this->lastTileLength ? this->tileLength : this->lastTileLength;
        // Align to 32 bytes
        uint32_t bufLen = (maxTile + alignNum - 1) / alignNum * alignNum;
        this->bufLen = bufLen;
        
        pipe.InitBuffer(inQueueX, 1, bufLen * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, bufLen * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, bufLen * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (this->blockOutLen == 0) return;
        
        for (uint32_t t = 0; t < this->tileNum; t++) {
            uint32_t curTileLen = (t < this->tileNum - 1) ? this->tileLength : this->lastTileLength;
            if (curTileLen == 0) continue;
            
            uint32_t alignNum = 8;
            uint32_t alignedLen = (curTileLen + alignNum - 1) / alignNum * alignNum;
            
            uint32_t outOffset = t * this->tileLength;
            
            // Initialize min with first slice along dim
            AscendC::LocalTensor<float> minLocal = outQueueZ.AllocTensor<float>();
            
            // Load first slice (d=0)
            // For each output element at position outOffset + i, 
            // the corresponding input index is:
            // outerIdx = (startIdx + outOffset + i) / innerSize
            // innerIdx = (startIdx + outOffset + i) % innerSize
            // inputIdx = outerIdx * dimSize * innerSize + d * innerSize + innerIdx
            
            if (this->innerSize == 1) {
                // Contiguous case: reduction over last-like dim with innerSize=1
                // Input for d-th slice: (startIdx + outOffset + i) * dimSize + d
                // Not contiguous in general, need to gather
                // Use scalar approach for non-contiguous
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                for (uint32_t i = 0; i < curTileLen; i++) {
                    uint32_t globalOutIdx = this->startIdx + outOffset + i;
                    uint32_t outerIdx = globalOutIdx; // since innerSize == 1
                    uint32_t baseIdx = outerIdx * this->dimSize;
                    float minVal = xGm.GetValue(baseIdx);
                    for (uint32_t d = 1; d < this->dimSize; d++) {
                        float val = xGm.GetValue(baseIdx + d);
                        if (val < minVal) minVal = val;
                    }
                    minLocal.SetValue(i, minVal);
                }
                tmpBuf.FreeTensor(tmpLocal);
            } else if (this->innerSize >= 8) {
                // innerSize is large enough for vectorized ops
                // For a given outerIdx, we can load innerSize elements contiguously
                // and do vector min
                
                // Process each output element
                // Group by outerIdx for better memory access
                AscendC::LocalTensor<float> sliceLocal = inQueueX.AllocTensor<float>();
                
                // First, load d=0 slice for all elements in this tile
                for (uint32_t i = 0; i < curTileLen; i++) {
                    uint32_t globalOutIdx = this->startIdx + outOffset + i;
                    uint32_t outerIdx = globalOutIdx / this->innerSize;
                    uint32_t innerIdx = globalOutIdx % this->innerSize;
                    uint32_t srcIdx = outerIdx * this->dimSize * this->innerSize + innerIdx;
                    float minVal = xGm.GetValue(srcIdx);
                    for (uint32_t d = 1; d < this->dimSize; d++) {
                        float val = xGm.GetValue(srcIdx + d * this->innerSize);
                        if (val < minVal) minVal = val;
                    }
                    minLocal.SetValue(i, minVal);
                }
                
                inQueueX.FreeTensor(sliceLocal);
            } else {
                // General scalar case
                for (uint32_t i = 0; i < curTileLen; i++) {
                    uint32_t globalOutIdx = this->startIdx + outOffset + i;
                    uint32_t outerIdx = globalOutIdx / this->innerSize;
                    uint32_t innerIdx = globalOutIdx % this->innerSize;
                    uint32_t baseIdx = outerIdx * this->dimSize * this->innerSize + innerIdx;
                    float minVal = xGm.GetValue(baseIdx);
                    for (uint32_t d = 1; d < this->dimSize; d++) {
                        float val = xGm.GetValue(baseIdx + d * this->innerSize);
                        if (val < minVal) minVal = val;
                    }
                    minLocal.SetValue(i, minVal);
                }
            }
            
            // Copy result out
            AscendC::DataCopy(yGm[outOffset], minLocal, alignedLen);
            outQueueZ.FreeTensor(minLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t blockOutLen;
    uint32_t startIdx;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t bufLen;
};

extern "C" __global__ __aicore__ void min_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinReduction op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dim, 
            tiling_data.dimSize, tiling_data.outerSize, tiling_data.innerSize, tiling_data.tileNum);
    op.Process();
}
