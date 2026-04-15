
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
constexpr float NEG_INF = -10000.0f;

class KernelCausalAttention {
public:
    __aicore__ inline KernelCausalAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t headDim,
                                 uint32_t numHeads, float scale)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->headDim = headDim;
        this->scale = scale;
        
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Number of instances (batch*heads) each block handles
        this->totalInstances = batchSize;
        this->instancesPerBlock = (totalInstances + blockNum - 1) / blockNum;
        this->startInstance = blockIdx * instancesPerBlock;
        this->endInstance = startInstance + instancesPerBlock;
        if (endInstance > totalInstances) endInstance = totalInstances;
        
        uint32_t instanceSize = seqLen * headDim;
        
        qGm.SetGlobalBuffer((__gm__ float *)q, batchSize * instanceSize);
        kGm.SetGlobalBuffer((__gm__ float *)k, batchSize * instanceSize);
        vGm.SetGlobalBuffer((__gm__ float *)v, batchSize * instanceSize);
        outGm.SetGlobalBuffer((__gm__ float *)out, batchSize * instanceSize);
        wsGm.SetGlobalBuffer((__gm__ float *)workspace, batchSize * seqLen * seqLen);
        
        // Align headDim and seqLen to 32 bytes (8 floats)
        this->headDimAligned = ((headDim + 7) / 8) * 8;
        this->seqLenAligned = ((seqLen + 7) / 8) * 8;
        
        // Buffer sizes
        uint32_t qRowSize = headDimAligned * sizeof(float);
        uint32_t kBlockSize = seqLen * headDimAligned * sizeof(float);
        uint32_t scoreRowSize = seqLenAligned * sizeof(float);
        uint32_t vBlockSize = seqLen * headDimAligned * sizeof(float);
        
        // We process one row of Q at a time
        // Need: 1 Q row, all K rows (tile if needed), score row, softmax temp, V rows, output row
        pipe.InitBuffer(inQueueQ, 1, qRowSize);
        pipe.InitBuffer(inQueueScore, 1, scoreRowSize);
        pipe.InitBuffer(outQueueOut, 1, headDimAligned * sizeof(float));
        pipe.InitBuffer(inQueueTemp, 1, scoreRowSize);  // for softmax intermediate
        pipe.InitBuffer(inQueueTemp2, 1, seqLenAligned * sizeof(float)); // for max/sum
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t inst = startInstance; inst < endInstance; inst++) {
            ProcessInstance(inst);
        }
    }

private:
    __aicore__ inline void ProcessInstance(uint32_t inst)
    {
        uint32_t qBase = inst * seqLen * headDim;
        uint32_t kBase = inst * seqLen * headDim;
        uint32_t vBase = inst * seqLen * headDim;
        uint32_t oBase = inst * seqLen * headDim;
        
        // For each query row
        for (uint32_t i = 0; i < seqLen; i++) {
            // Step 1: Compute attention scores for row i: score[j] = sum_d(Q[i,d]*K[j,d]) * scale
            AscendC::LocalTensor<float> scoreLocal = inQueueScore.AllocTensor<float>();
            
            // Initialize scores to 0
            AscendC::Duplicate(scoreLocal, 0.0f, seqLenAligned);
            
            // Compute dot products manually
            AscendC::LocalTensor<float> qRow = inQueueQ.AllocTensor<float>();
            
            // Load Q[i,:] 
            AscendC::DataCopy(qRow, qGm[qBase + i * headDim], headDimAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            // Scale Q
            AscendC::Muls(qRow, qRow, scale, headDimAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            AscendC::LocalTensor<float> tempLocal = inQueueTemp.AllocTensor<float>();
            
            // For each key j, compute dot product
            for (uint32_t j = 0; j <= i; j++) {
                // Load K[j,:]
                AscendC::DataCopy(tempLocal, kGm[kBase + j * headDim], headDimAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
                
                // Multiply element-wise
                AscendC::Mul(tempLocal, qRow, tempLocal, headDimAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
                
                // Sum reduction
                float dotVal = 0.0f;
                for (uint32_t d = 0; d < headDim; d++) {
                    dotVal += tempLocal.GetValue(d);
                }
                scoreLocal.SetValue(j, dotVal);
            }
            
            // Apply causal mask: set positions j > i to -inf
            for (uint32_t j = i + 1; j < seqLen; j++) {
                scoreLocal.SetValue(j, NEG_INF);
            }
            // Also set padding to -inf
            for (uint32_t j = seqLen; j < seqLenAligned; j++) {
                scoreLocal.SetValue(j, NEG_INF);
            }
            
            AscendC::PipeBarrier<PIPE_ALL>();
            
            // Step 2: Softmax over scores
            // Find max
            AscendC::LocalTensor<float> temp2Local = inQueueTemp2.AllocTensor<float>();
            
            float maxVal = NEG_INF;
            for (uint32_t j = 0; j < seqLen; j++) {
                float sv = scoreLocal.GetValue(j);
                if (sv > maxVal) maxVal = sv;
            }
            
            // Subtract max and exp
            AscendC::Adds(scoreLocal, scoreLocal, -maxVal, seqLenAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(scoreLocal, scoreLocal, seqLenAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            // Zero out padding positions
            for (uint32_t j = seqLen; j < seqLenAligned; j++) {
                scoreLocal.SetValue(j, 0.0f);
            }
            
            // Sum
            float sumVal = 0.0f;
            for (uint32_t j = 0; j < seqLen; j++) {
                sumVal += scoreLocal.GetValue(j);
            }
            if (sumVal < 1e-12f) sumVal = 1e-12f;
            float invSum = 1.0f / sumVal;
            
            AscendC::Muls(scoreLocal, scoreLocal, invSum, seqLenAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            // Step 3: Compute output[i,:] = sum_j score[j] * V[j,:]
            AscendC::LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
            AscendC::Duplicate(outLocal, 0.0f, headDimAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            for (uint32_t j = 0; j <= i; j++) {
                float w = scoreLocal.GetValue(j);
                if (w < 1e-12f) continue;
                
                // Load V[j,:]
                AscendC::DataCopy(tempLocal, vGm[vBase + j * headDim], headDimAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
                
                // outLocal += w * V[j,:]
                AscendC::Muls(tempLocal, tempLocal, w, headDimAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(outLocal, outLocal, tempLocal, headDimAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            
            // Store output row
            AscendC::DataCopy(outGm[oBase + i * headDim], outLocal, headDimAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            
            inQueueQ.FreeTensor(qRow);
            inQueueScore.FreeTensor(scoreLocal);
            inQueueTemp.FreeTensor(tempLocal);
            inQueueTemp2.FreeTensor(temp2Local);
            outQueueOut.FreeTensor(outLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueScore;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueTemp;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueTemp2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueOut;
    
    AscendC::GlobalTensor<float> qGm;
    AscendC::GlobalTensor<float> kGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::GlobalTensor<float> wsGm;
    
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t headDim;
    uint32_t totalInstances;
    uint32_t instancesPerBlock;
    uint32_t startInstance;
    uint32_t endInstance;
    uint32_t headDimAligned;
    uint32_t seqLenAligned;
    float scale;
};

extern "C" __global__ __aicore__ void causal_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCausalAttention op;
    op.Init(q, k, v, out, workspace,
            tiling_data.batchSize, tiling_data.seqLen, tiling_data.headDim,
            tiling_data.numHeads, tiling_data.scale);
    op.Process();
}
