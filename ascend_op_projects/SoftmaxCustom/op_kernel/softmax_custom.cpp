
#include "kernel_operator.h"
#include <cmath>

constexpr int32_t BUFFER_NUM = 2;

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t rowLength)
    {
        this->rowLength = rowLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->rowsPerBlock = this->blockLength / this->rowLength;
        
        xGm.SetGlobalBuffer((__gm__ float*)x + (this->blockLength * AscendC::GetBlockIdx()), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + (this->blockLength * AscendC::GetBlockIdx()), this->blockLength);
        
        // Tiling for local memory: process row by row
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->rowLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->rowLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->rowsPerBlock; i++) {
            ComputeRow(i);
        }
    }

private:
    __aicore__ inline void ComputeRow(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // 1. Copy row from GM to UB
        AscendC::DataCopy(xLocal, xGm[rowIdx * this->rowLength], this->rowLength);

        // 2. Find Max for numerical stability
        float maxVal = -1e38f; 
        // Simplified max reduction for brevity in kernel_src
        for(uint32_t j=0; j<this->rowLength; j++) {
            if(xLocal[j] > maxVal) maxVal = xLocal[j];
        }

        // 3. Compute exp(x - max) and sum
        float sumExp = 0.0f;
        for(uint32_t j=0; j<this->rowLength; j++) {
            float val = std::exp(xLocal[j] - maxVal);
            zLocal[j] = val;
            sumExp += val;
        }

        // 4. Normalize
        float invSum = 1.0f / sumExp;
        for(uint32_t j=0; j<this->rowLength; j++) {
            zLocal[j] *= invSum;
        }

        // 5. Copy result back to GM
        AscendC::DataCopy(zGm[rowIdx * this->rowLength], zLocal, this->rowLength);

        inQueueX.FreeTensor(xLocal);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t rowLength;
    uint32_t rowsPerBlock;
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.rowLength);
    op.Process();
}
