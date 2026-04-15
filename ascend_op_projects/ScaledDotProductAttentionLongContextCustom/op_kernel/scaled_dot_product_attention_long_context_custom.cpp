
#include "kernel_operator.h"

// Flash-attention style tiled kernel for scaled dot-product attention
// Q, K, V: [batch, seqLen, dModel]
// Output: softmax(Q @ K^T / sqrt(dModel)) @ V

constexpr int32_t BUFFER_NUM = 1;

class KernelScaledDotProductAttention {
public:
    __aicore__ inline KernelScaledDotProductAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t dModel,
                                 uint32_t blockQ, uint32_t blockKV)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->dModel = dModel;
        this->blockQ = blockQ;
        this->blockKV = blockKV;
        this->scale = 1.0f / AscendC::Sqrt(static_cast<float>(dModel));

        uint32_t totalQBlocks = (seqLen + blockQ - 1) / blockQ;
        uint32_t totalTasks = batchSize * totalQBlocks;
        uint32_t numCores = AscendC::GetBlockNum();
        uint32_t coreId = AscendC::GetBlockIdx();

        this->tasksPerCore = (totalTasks + numCores - 1) / numCores;
        this->taskStart = coreId * this->tasksPerCore;
        this->taskEnd = taskStart + tasksPerCore;
        if (this->taskEnd > totalTasks) this->taskEnd = totalTasks;
        this->totalQBlocks = totalQBlocks;

        uint64_t batchStride = (uint64_t)seqLen * dModel;
        qGm.SetGlobalBuffer((__gm__ float*)q, batchSize * batchStride);
        kGm.SetGlobalBuffer((__gm__ float*)k, batchSize * batchStride);
        vGm.SetGlobalBuffer((__gm__ float*)v, batchSize * batchStride);
        outGm.SetGlobalBuffer((__gm__ float*)output, batchSize * batchStride);
        wsGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize * batchStride);

        // Allocate buffers for tile computation
        // We need: qTile[blockQ * dModel_chunk], kTile[blockKV * dModel_chunk], 
        // scores[blockQ * blockKV], vTile[blockKV * dModel_chunk], oTile[blockQ * dModel_chunk]
        // Due to UB size limits, we process dModel in chunks
        
        // Calculate dModel chunk size based on available UB
        // Each tile needs memory, we keep it conservative
        this->dChunk = 64; // process dModel in chunks of 64
        if (this->dChunk > dModel) this->dChunk = dModel;

        uint32_t qTileSize = blockQ * dChunk;
        uint32_t kTileSize = blockKV * dChunk;
        uint32_t scoreTileSize = blockQ * blockKV;
        uint32_t vTileSize = blockKV * dChunk;
        uint32_t oTileSize = blockQ * dChunk;

        // Align sizes to 32 bytes (8 floats)
        auto align = [](uint32_t x) -> uint32_t { return ((x + 7) / 8) * 8; };
        qTileSize = align(qTileSize);
        kTileSize = align(kTileSize);
        scoreTileSize = align(scoreTileSize);
        vTileSize = align(vTileSize);
        oTileSize = align(oTileSize);

        this->qTileSize = qTileSize;
        this->kTileSize = kTileSize;
        this->scoreTileSize = scoreTileSize;
        this->vTileSize = vTileSize;
        this->oTileSize = oTileSize;

        pipe.InitBuffer(qBuf, 1, qTileSize * sizeof(float));
        pipe.InitBuffer(kBuf, 1, kTileSize * sizeof(float));
        pipe.InitBuffer(scoreBuf, 1, scoreTileSize * sizeof(float));
        pipe.InitBuffer(vBuf, 1, vTileSize * sizeof(float));
        pipe.InitBuffer(oBuf, 1, oTileSize * sizeof(float));
        // buffers for softmax: rowMax[blockQ], rowSum[blockQ], prevMax[blockQ], prevSum[blockQ]
        uint32_t rowSize = align(blockQ);
        this->rowSize = rowSize;
        pipe.InitBuffer(rowMaxBuf, 1, rowSize * sizeof(float));
        pipe.InitBuffer(rowSumBuf, 1, rowSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, scoreTileSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = taskStart; t < taskEnd; t++) {
            uint32_t bIdx = t / totalQBlocks;
            uint32_t qBlockIdx = t % totalQBlocks;
            ProcessOneQBlock(bIdx, qBlockIdx);
        }
    }

private:
    __aicore__ inline void ProcessOneQBlock(uint32_t bIdx, uint32_t qBlockIdx)
    {
        uint32_t qStart = qBlockIdx * blockQ;
        uint32_t qEnd = qStart + blockQ;
        if (qEnd > seqLen) qEnd = seqLen;
        uint32_t qLen = qEnd - qStart;
        
        uint64_t batchOff = (uint64_t)bIdx * seqLen * dModel;
        
        uint32_t numKVBlocks = (seqLen + blockKV - 1) / blockKV;
        uint32_t numDChunks = (dModel + dChunk - 1) / dChunk;
        
        // Initialize row max and row sum
        AscendC::LocalTensor<float> rowMax = rowMaxBuf.Get<float>();
        AscendC::LocalTensor<float> rowSum = rowSumBuf.Get<float>();
        
        // Set rowMax to -inf, rowSum to 0
        AscendC::Duplicate(rowMax, -1e30f, rowSize);
        AscendC::Duplicate(rowSum, 0.0f, rowSize);
        
        // For each KV block, we accumulate output in workspace
        // First zero out the output accumulator in workspace
        // ws layout: use outGm directly, init to zero
        // We'll accumulate per d-chunk into outGm
        
        // Zero the output for this q-block
        AscendC::LocalTensor<float> oLocal = oBuf.Get<float>();
        for (uint32_t dc = 0; dc < numDChunks; dc++) {
            uint32_t dStart = dc * dChunk;
            uint32_t dEnd = dStart + dChunk;
            if (dEnd > dModel) dEnd = dModel;
            uint32_t dLen = dEnd - dStart;
            uint32_t dLenAligned = ((dLen + 7) / 8) * 8;
            
            for (uint32_t qi = 0; qi < qLen; qi++) {
                AscendC::Duplicate(oLocal[qi * dLenAligned], 0.0f, dLenAligned);
            }
            // Write zeros to workspace
            for (uint32_t qi = 0; qi < qLen; qi++) {
                AscendC::DataCopy(wsGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart],
                                  oLocal[qi * dLenAligned], dLenAligned);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        
        // Iterate over KV blocks
        for (uint32_t kvb = 0; kvb < numKVBlocks; kvb++) {
            uint32_t kvStart = kvb * blockKV;
            uint32_t kvEnd = kvStart + blockKV;
            if (kvEnd > seqLen) kvEnd = seqLen;
            uint32_t kvLen = kvEnd - kvStart;
            
            // Compute scores[qLen x kvLen] = sum over d-chunks of Q_chunk @ K_chunk^T
            AscendC::LocalTensor<float> scores = scoreBuf.Get<float>();
            // Zero scores
            uint32_t scoreTotal = ((qLen * kvLen + 7) / 8) * 8;
            AscendC::Duplicate(scores, 0.0f, scoreTileSize);
            
            // Accumulate Q @ K^T over d-chunks
            for (uint32_t dc = 0; dc < numDChunks; dc++) {
                uint32_t dStart = dc * dChunk;
                uint32_t dEnd2 = dStart + dChunk;
                if (dEnd2 > dModel) dEnd2 = dModel;
                uint32_t dLen = dEnd2 - dStart;
                uint32_t dLenAligned = ((dLen + 7) / 8) * 8;
                
                // Load Q tile [qLen x dLen]
                AscendC::LocalTensor<float> qLocal = qBuf.Get<float>();
                for (uint32_t qi = 0; qi < qLen; qi++) {
                    AscendC::DataCopy(qLocal[qi * dLenAligned],
                                      qGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart],
                                      dLenAligned);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                
                // Load K tile [kvLen x dLen]
                AscendC::LocalTensor<float> kLocal = kBuf.Get<float>();
                for (uint32_t ki = 0; ki < kvLen; ki++) {
                    AscendC::DataCopy(kLocal[ki * dLenAligned],
                                      kGm[batchOff + (uint64_t)(kvStart + ki) * dModel + dStart],
                                      dLenAligned);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                
                // Compute partial dot products and accumulate into scores
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
                for (uint32_t qi = 0; qi < qLen; qi++) {
                    for (uint32_t ki = 0; ki < kvLen; ki++) {
                        // dot product of qLocal[qi*dLenAligned : +dLenAligned] and kLocal[ki*dLenAligned : +dLenAligned]
                        AscendC::Mul(tmpLocal, qLocal[qi * dLenAligned], kLocal[ki * dLenAligned], dLenAligned);
                        // reduce sum
                        float dotVal = 0.0f;
                        for (uint32_t d = 0; d < dLen; d++) {
                            dotVal += tmpLocal.GetValue(d);
                        }
                        float cur = scores.GetValue(qi * kvLen + ki);
                        scores.SetValue(qi * kvLen + ki, cur + dotVal);
                    }
                }
            }
            
            // Scale scores
            AscendC::Muls(scores, scores, this->scale, scoreTileSize);
            
            // Online softmax update
            // For each query row qi:
            //   newMax = max(rowMax[qi], max(scores[qi, :]))
            //   correction = exp(rowMax[qi] - newMax)
            //   rowSum[qi] = rowSum[qi] * correction + sum(exp(scores[qi,:] - newMax))
            //   rowMax[qi] = newMax
            //   Rescale existing output accumulator by correction
            //   Add exp(scores[qi,ki] - newMax) * V[ki,:] to output accumulator
            
            for (uint32_t qi = 0; qi < qLen; qi++) {
                float oldMax = rowMax.GetValue(qi);
                float oldSum = rowSum.GetValue(qi);
                
                // Find max of scores[qi, :]
                float newMax = oldMax;
                for (uint32_t ki = 0; ki < kvLen; ki++) {
                    float s = scores.GetValue(qi * kvLen + ki);
                    if (s > newMax) newMax = s;
                }
                
                float correction = AscendC::Exp(oldMax - newMax);
                
                // Compute exp(scores - newMax) and their sum
                float expSum = 0.0f;
                for (uint32_t ki = 0; ki < kvLen; ki++) {
                    float s = scores.GetValue(qi * kvLen + ki);
                    float e = AscendC::Exp(s - newMax);
                    scores.SetValue(qi * kvLen + ki, e);
                    expSum += e;
                }
                
                float newSum = oldSum * correction + expSum;
                rowMax.SetValue(qi, newMax);
                rowSum.SetValue(qi, newSum);
                
                // Update output: rescale old output and add new contribution
                // Process in d-chunks
                for (uint32_t dc = 0; dc < numDChunks; dc++) {
                    uint32_t dStart = dc * dChunk;
                    uint32_t dEnd2 = dStart + dChunk;
                    if (dEnd2 > dModel) dEnd2 = dModel;
                    uint32_t dLen = dEnd2 - dStart;
                    uint32_t dLenAligned = ((dLen + 7) / 8) * 8;
                    
                    // Load current output accumulator
                    AscendC::LocalTensor<float> oTile = oBuf.Get<float>();
                    AscendC::DataCopy(oTile, wsGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart], dLenAligned);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                    
                    // Rescale by correction
                    AscendC::Muls(oTile, oTile, correction, dLenAligned);
                    
                    // Load V rows and accumulate
                    AscendC::LocalTensor<float> vLocal = vBuf.Get<float>();
                    for (uint32_t ki = 0; ki < kvLen; ki++) {
                        float attnWeight = scores.GetValue(qi * kvLen + ki);
                        AscendC::DataCopy(vLocal, vGm[batchOff + (uint64_t)(kvStart + ki) * dModel + dStart], dLenAligned);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                        // oTile += attnWeight * vLocal
                        AscendC::LocalTensor<float> tmpD = tmpBuf.Get<float>();
                        AscendC::Muls(tmpD, vLocal, attnWeight, dLenAligned);
                        AscendC::Add(oTile, oTile, tmpD, dLenAligned);
                    }
                    
                    // Store back
                    AscendC::DataCopy(wsGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart], oTile, dLenAligned);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                }
            }
        }
        
        // Final normalization: divide by rowSum
        uint32_t numDChunks2 = (dModel + dChunk - 1) / dChunk;
        for (uint32_t qi = 0; qi < qLen; qi++) {
            float s = rowSum.GetValue(qi);
            float invS = (s > 0.0f) ? (1.0f / s) : 0.0f;
            
            for (uint32_t dc = 0; dc < numDChunks2; dc++) {
                uint32_t dStart = dc * dChunk;
                uint32_t dEnd2 = dStart + dChunk;
                if (dEnd2 > dModel) dEnd2 = dModel;
                uint32_t dLen = dEnd2 - dStart;
                uint32_t dLenAligned = ((dLen + 7) / 8) * 8;
                
                AscendC::LocalTensor<float> oTile = oBuf.Get<float>();
                AscendC::DataCopy(oTile, wsGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart], dLenAligned);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                
                AscendC::Muls(oTile, oTile, invS, dLenAligned);
                
                AscendC::DataCopy(outGm[batchOff + (uint64_t)(qStart + qi) * dModel + dStart], oTile, dLenAligned);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf, kBuf, scoreBuf, vBuf, oBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxBuf, rowSumBuf, tmpBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm, wsGm;
    uint32_t batchSize, seqLen, dModel;
    uint32_t blockQ, blockKV;
    uint32_t dChunk;
    uint32_t totalQBlocks;
    uint32_t tasksPerCore, taskStart, taskEnd;
    float scale;
    uint32_t qTileSize, kTileSize, scoreTileSize, vTileSize, oTileSize, rowSize;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_long_context_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaledDotProductAttention op;
    op.Init(q, k, v, output, workspace,
            tiling_data.batchSize, tiling_data.seqLen, tiling_data.dModel,
            tiling_data.blockQ, tiling_data.blockKV);
    op.Process();
}
