
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelLinearAttention {
public:
    __aicore__ inline KernelLinearAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t dModel)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->dModel = dModel;
        
        // Each block handles one batch element
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= batchSize) return;
        
        uint32_t batchOffset = blockIdx * seqLen * dModel;

        qGm.SetGlobalBuffer((__gm__ float *)q + batchOffset, seqLen * dModel);
        kGm.SetGlobalBuffer((__gm__ float *)k + batchOffset, seqLen * dModel);
        vGm.SetGlobalBuffer((__gm__ float *)v + batchOffset, seqLen * dModel);
        outGm.SetGlobalBuffer((__gm__ float *)out + batchOffset, seqLen * dModel);
        
        // workspace for KV matrix: dModel x dModel per batch
        kvGm.SetGlobalBuffer((__gm__ float *)workspace + blockIdx * dModel * dModel, dModel * dModel);
        
        // Align tile sizes to 32 bytes (8 floats)
        this->dModelAligned = (dModel + 7) / 8 * 8;
        
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(inQueueQ, BUFFER_NUM, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(kvRowBuf, 1, this->dModelAligned * sizeof(float));
        pipe.InitBuffer(epsBuf, 1, this->dModelAligned * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= batchSize) return;
        
        // Step 1: Apply ReLU + eps to Q and K, compute KV = K^T * V
        // KV is dModel x dModel
        // KV[d][v] = sum_k K_[k][d] * V[k][v]
        
        // Initialize KV to zero
        for (uint32_t d = 0; d < dModel; d++) {
            AscendC::LocalTensor<float> kvRow = kvRowBuf.Get<float>();
            AscendC::Duplicate(kvRow, (float)0.0f, this->dModelAligned);
            AscendC::DataCopy(kvGm[d * dModel], kvRow, dModel);
        }
        
        // Initialize eps buffer
        {
            AscendC::LocalTensor<float> epsLocal = epsBuf.Get<float>();
            AscendC::Duplicate(epsLocal, 1e-6f, this->dModelAligned);
        }
        
        // Compute KV: for each sequence position k, accumulate outer product
        for (uint32_t s = 0; s < seqLen; s++) {
            // Load K row and apply ReLU + eps
            AscendC::LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
            AscendC::DataCopy(kLocal, kGm[s * dModel], this->dModelAligned);
            inQueueK.EnQue(kLocal);
            kLocal = inQueueK.DeQue<float>();
            
            // ReLU
            AscendC::Maxs(kLocal, kLocal, (float)0.0f, this->dModelAligned);
            // Add eps
            AscendC::LocalTensor<float> epsLocal = epsBuf.Get<float>();
            AscendC::Add(kLocal, kLocal, epsLocal, this->dModelAligned);
            
            // Load V row
            AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
            AscendC::DataCopy(vLocal, vGm[s * dModel], this->dModelAligned);
            inQueueV.EnQue(vLocal);
            vLocal = inQueueV.DeQue<float>();
            
            // For each dimension d in K, accumulate k_val * V into KV[d][:]
            for (uint32_t d = 0; d < dModel; d++) {
                float kVal = kLocal.GetValue(d);
                if (kVal == 0.0f) continue;
                
                AscendC::LocalTensor<float> kvRow = kvRowBuf.Get<float>();
                AscendC::DataCopy(kvRow, kvGm[d * dModel], this->dModelAligned);
                
                AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
                AscendC::Muls(tmp, vLocal, kVal, this->dModelAligned);
                AscendC::Add(kvRow, kvRow, tmp, this->dModelAligned);
                
                AscendC::DataCopy(kvGm[d * dModel], kvRow, dModel);
            }
            
            inQueueK.FreeTensor(kLocal);
            inQueueV.FreeTensor(vLocal);
        }
        
        // Step 2: Compute output = Q_ * KV
        // out[q][v] = sum_d Q_[q][d] * KV[d][v]
        for (uint32_t s = 0; s < seqLen; s++) {
            // Load Q row and apply ReLU + eps
            AscendC::LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
            AscendC::DataCopy(qLocal, qGm[s * dModel], this->dModelAligned);
            inQueueQ.EnQue(qLocal);
            qLocal = inQueueQ.DeQue<float>();
            
            AscendC::Maxs(qLocal, qLocal, (float)0.0f, this->dModelAligned);
            AscendC::LocalTensor<float> epsLocal = epsBuf.Get<float>();
            AscendC::Add(qLocal, qLocal, epsLocal, this->dModelAligned);
            
            // Compute out row = sum_d q[d] * KV[d][:]
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            AscendC::Duplicate(outLocal, (float)0.0f, this->dModelAligned);
            
            for (uint32_t d = 0; d < dModel; d++) {
                float qVal = qLocal.GetValue(d);
                if (qVal == 0.0f) continue;
                
                AscendC::LocalTensor<float> kvRow = kvRowBuf.Get<float>();
                AscendC::DataCopy(kvRow, kvGm[d * dModel], this->dModelAligned);
                
                AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
                AscendC::Muls(tmp, kvRow, qVal, this->dModelAligned);
                AscendC::Add(outLocal, outLocal, tmp, this->dModelAligned);
            }
            
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<float>();
            AscendC::DataCopy(outGm[s * dModel], outLocal, dModel);
            
            inQueueQ.FreeTensor(qLocal);
            outQueue.FreeTensor(outLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueK, inQueueV, inQueueQ;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf, kvRowBuf, epsBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm, kvGm;
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t dModel;
    uint32_t dModelAligned;
};

extern "C" __global__ __aicore__ void linear_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLinearAttention op;
    op.Init(q, k, v, out, workspace, tiling_data.batchSize, tiling_data.seqLen, tiling_data.dModel);
    op.Process();
}
