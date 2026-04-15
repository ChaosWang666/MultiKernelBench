
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSumReduction {
public:
    __aicore__ inline KernelSumReduction() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dim,
                                 uint32_t shape0, uint32_t shape1, uint32_t shape2, uint32_t tileNum)
    {
        this->dim = dim;
        this->shape0 = shape0;
        this->shape1 = shape1;
        this->shape2 = shape2;

        // For 3D tensor of shape [shape0, shape1, shape2], reducing over dim:
        // dim=0: outerSize=1, reduceSize=shape0, innerSize=shape1*shape2
        // dim=1: outerSize=shape0, reduceSize=shape1, innerSize=shape2
        // dim=2: outerSize=shape0*shape1, reduceSize=shape2, innerSize=1
        if (dim == 0) {
            this->outerSize = 1;
            this->reduceSize = shape0;
            this->innerSize = shape1 * shape2;
        } else if (dim == 1) {
            this->outerSize = shape0;
            this->reduceSize = shape1;
            this->innerSize = shape2;
        } else {
            this->outerSize = shape0 * shape1;
            this->reduceSize = shape2;
            this->innerSize = 1;
        }

        // Total output elements = outerSize * innerSize
        uint32_t outputTotal = this->outerSize * this->innerSize;
        
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Each block processes a chunk of output elements
        uint32_t perBlock = (outputTotal + blockNum - 1) / blockNum;
        this->startOut = blockIdx * perBlock;
        this->endOut = this->startOut + perBlock;
        if (this->endOut > outputTotal) this->endOut = outputTotal;
        if (this->startOut > outputTotal) this->startOut = outputTotal;
        
        this->outputCount = this->endOut - this->startOut;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, outputTotal);

        // Allocate buffer for one reduction stripe
        uint32_t alignedReduce = ((this->reduceSize + 7) / 8) * 8;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedReduce * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, 8 * sizeof(float));
        // work buffer for ReduceSum
        pipe.InitBuffer(workBuf, 1, alignedReduce * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->outputCount; i++) {
            uint32_t outIdx = this->startOut + i;
            uint32_t outerIdx = outIdx / this->innerSize;
            uint32_t innerIdx = outIdx % this->innerSize;
            
            // Gather reduceSize elements and sum
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            uint32_t alignedReduce = ((this->reduceSize + 7) / 8) * 8;
            
            // Use dimension-specific gathering
            if (dim == 1) {
                // x layout: [shape0, shape1, shape2]
                // For fixed outerIdx and innerIdx, gather x[outerIdx, r, innerIdx] for r=0..reduceSize-1
                // stride between consecutive r values = innerSize (= shape2)
                uint32_t baseOffset = outerIdx * this->reduceSize * this->innerSize + innerIdx;
                for (uint32_t r = 0; r < this->reduceSize; r++) {
                    uint32_t srcIdx = baseOffset + r * this->innerSize;
                    // We need to use DataCopy for single element - copy 1 element
                    // But DataCopy requires alignment. Use scalar approach via pipe.
                    // For simplicity, use SetValue from GM through a different approach.
                    float val = *(((__gm__ float*)xGm.GetPhyAddr()) + srcIdx);
                    xLocal.SetValue(r, val);
                }
                // Pad remaining with 0
                for (uint32_t r = this->reduceSize; r < alignedReduce; r++) {
                    xLocal.SetValue(r, 0.0f);
                }
            } else if (dim == 0) {
                // x layout: [shape0, shape1, shape2]
                // gather x[r, outerIdx_notused, innerIdx] but outerSize=1 so innerIdx is the linear index in shape1*shape2
                // stride = shape1*shape2
                uint32_t baseOffset = innerIdx;
                uint32_t stride = this->shape1 * this->shape2;
                for (uint32_t r = 0; r < this->reduceSize; r++) {
                    float val = *(((__gm__ float*)xGm.GetPhyAddr()) + baseOffset + r * stride);
                    xLocal.SetValue(r, val);
                }
                for (uint32_t r = this->reduceSize; r < alignedReduce; r++) {
                    xLocal.SetValue(r, 0.0f);
                }
            } else {
                // dim == 2
                // x layout: [shape0, shape1, shape2]
                // outerIdx = shape0*shape1 index, gather contiguous shape2 elements
                uint32_t baseOffset = outerIdx * this->reduceSize;
                // Contiguous: can DataCopy
                AscendC::DataCopy(xLocal, xGm[baseOffset], alignedReduce);
                // Zero out padding
                for (uint32_t r = this->reduceSize; r < alignedReduce; r++) {
                    xLocal.SetValue(r, 0.0f);
                }
            }
            inQueueX.EnQue(xLocal);
            
            // Compute sum
            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            AscendC::LocalTensor<float> wLocal = workBuf.Get<float>();
            
            AscendC::ReduceSum(zLocal, xIn, wLocal, alignedReduce);
            
            outQueueZ.EnQue(zLocal);
            inQueueX.FreeTensor(xIn);
            
            // Copy out single value
            AscendC::LocalTensor<float> zOut = outQueueZ.DeQue<float>();
            // Write single float to GM
            // Use DataCopy with minimum alignment
            AscendC::DataCopy(yGm[outIdx * 8], zOut, 8);
            outQueueZ.FreeTensor(zOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t dim;
    uint32_t shape0, shape1, shape2;
    uint32_t outerSize, reduceSize, innerSize;
    uint32_t startOut, endOut, outputCount;
};

// Optimized kernel for dim=1 case (the common case): reduce along dim1 for [B, D1, D2]
// Each block handles a subset of (batch, d2) pairs
// For each pair, sum D1 contiguous-strided elements
class KernelSumReduceDim1 {
public:
    __aicore__ inline KernelSumReduceDim1() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t shape0, uint32_t shape1, uint32_t shape2)
    {
        this->shape0 = shape0;
        this->shape1 = shape1;
        this->shape2 = shape2;
        
        uint32_t outputTotal = shape0 * shape2; // one output per (batch, d2) pair
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        uint32_t perBlock = (outputTotal + blockNum - 1) / blockNum;
        this->startOut = blockIdx * perBlock;
        this->endOut = this->startOut + perBlock;
        if (this->endOut > outputTotal) this->endOut = outputTotal;
        if (this->startOut > outputTotal) this->startOut = outputTotal;
        this->outputCount = this->endOut - this->startOut;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, shape0 * shape1 * shape2);
        yGm.SetGlobalBuffer((__gm__ float *)y, outputTotal);
        
        uint32_t alignedReduce = ((shape1 + 7) / 8) * 8;
        pipe.InitBuffer(inQueueX, 1, alignedReduce * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, 8 * sizeof(float));
        pipe.InitBuffer(workBuf, 1, alignedReduce * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        uint32_t alignedReduce = ((shape1 + 7) / 8) * 8;
        for (uint32_t i = 0; i < this->outputCount; i++) {
            uint32_t outIdx = this->startOut + i;
            uint32_t batchIdx = outIdx / shape2;
            uint32_t d2Idx = outIdx % shape2;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            
            // Gather shape1 elements: x[batchIdx, r, d2Idx]
            uint32_t baseOffset = batchIdx * shape1 * shape2 + d2Idx;
            for (uint32_t r = 0; r < shape1; r++) {
                float val = *(((__gm__ float*)xGm.GetPhyAddr()) + baseOffset + r * shape2);
                xLocal.SetValue(r, val);
            }
            for (uint32_t r = shape1; r < alignedReduce; r++) {
                xLocal.SetValue(r, 0.0f);
            }
            inQueueX.EnQue(xLocal);
            
            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            AscendC::LocalTensor<float> wLocal = workBuf.Get<float>();
            AscendC::ReduceSum(zLocal, xIn, wLocal, alignedReduce);
            outQueueZ.EnQue(zLocal);
            inQueueX.FreeTensor(xIn);
            
            AscendC::LocalTensor<float> zOut = outQueueZ.DeQue<float>();
            // Write result - we need to write single float but DataCopy needs alignment
            // We'll use a trick: write to a padded output buffer
            AscendC::DataCopy(yGm[outIdx * 8], zOut, 8);
            outQueueZ.FreeTensor(zOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t shape0, shape1, shape2;
    uint32_t startOut, endOut, outputCount;
};

extern "C" __global__ __aicore__ void sum_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSumReduction op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dim,
            tiling_data.shape0, tiling_data.shape1, tiling_data.shape2, tiling_data.tileNum);
    op.Process();
}
