
#include "kernel_operator.h"

class KernelEmbedding {
public:
    __aicore__ inline KernelEmbedding() {}
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR indices, GM_ADDR output,
                                 uint32_t numIndices, uint32_t embeddingDim)
    {
        this->numIndices = numIndices;
        this->embeddingDim = embeddingDim;
        
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Distribute indices across blocks
        this->indicesPerBlock = (numIndices + blockNum - 1) / blockNum;
        this->startIdx = blockIdx * this->indicesPerBlock;
        this->endIdx = startIdx + indicesPerBlock;
        if (this->endIdx > numIndices) {
            this->endIdx = numIndices;
        }
        
        // Align embeddingDim to 8 floats (32 bytes) for DataCopy
        this->alignedDim = ((embeddingDim + 7) / 8) * 8;
        
        weightGm.SetGlobalBuffer((__gm__ float *)weight, 100000 * embeddingDim);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, numIndices);
        outputGm.SetGlobalBuffer((__gm__ float *)output, numIndices * embeddingDim);
        
        pipe.InitBuffer(indicesQueue, 1, this->indicesPerBlock * sizeof(int32_t) < 32 ? 32 : ((this->indicesPerBlock * sizeof(int32_t) + 31) / 32) * 32);
        pipe.InitBuffer(dataQueue, 1, this->alignedDim * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (this->startIdx >= this->numIndices) return;
        
        uint32_t count = this->endIdx - this->startIdx;
        
        // Copy indices for this block
        uint32_t indicesCopyLen = ((count * sizeof(int32_t) + 31) / 32) * 32 / sizeof(int32_t);
        AscendC::LocalTensor<int32_t> indicesLocal = indicesQueue.AllocTensor<int32_t>();
        AscendC::DataCopy(indicesLocal, indicesGm[this->startIdx], indicesCopyLen);
        indicesQueue.EnQue(indicesLocal);
        indicesLocal = indicesQueue.DeQue<int32_t>();
        
        for (uint32_t i = 0; i < count; i++) {
            int32_t idx = indicesLocal.GetValue(i);
            
            AscendC::LocalTensor<float> dataLocal = dataQueue.AllocTensor<float>();
            // Copy embedding row
            AscendC::DataCopy(dataLocal, weightGm[idx * this->embeddingDim], this->alignedDim);
            dataQueue.EnQue(dataLocal);
            dataLocal = dataQueue.DeQue<float>();
            
            // Write to output
            uint32_t outOffset = (this->startIdx + i) * this->embeddingDim;
            AscendC::DataCopy(outputGm[outOffset], dataLocal, this->alignedDim);
            dataQueue.FreeTensor(dataLocal);
        }
        
        indicesQueue.FreeTensor(indicesLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> indicesQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> dataQueue;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t numIndices;
    uint32_t embeddingDim;
    uint32_t alignedDim;
    uint32_t indicesPerBlock;
    uint32_t startIdx;
    uint32_t endIdx;
};

extern "C" __global__ __aicore__ void embedding_custom(GM_ADDR weight, GM_ADDR indices, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelEmbedding op;
    op.Init(weight, indices, output, tiling_data.numIndices, tiling_data.embeddingDim);
    op.Process();
}
