
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelAttention {
public:
    __aicore__ inline KernelAttention() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR output,
                                 uint32_t batchSize, uint32_t qLen, uint32_t kvLen, uint32_t dModel, float scale)
    {
        this->batchSize = batchSize;
        this->qLen = qLen;
        this->kvLen = kvLen;
        this->dModel = dModel;
        this->scale = scale;

        uint32_t totalQueries = batchSize * qLen;
        uint32_t numBlocks = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        // Distribute queries across blocks
        uint32_t queriesPerBlock = (totalQueries + numBlocks - 1) / numBlocks;
        this->startQuery = blockIdx * queriesPerBlock;
        this->endQuery = startQuery + queriesPerBlock;
        if (this->endQuery > totalQueries) this->endQuery = totalQueries;

        qGm.SetGlobalBuffer((__gm__ float*)query, batchSize * qLen * dModel);
        kGm.SetGlobalBuffer((__gm__ float*)key_cache, batchSize * kvLen * dModel);
        vGm.SetGlobalBuffer((__gm__ float*)value_cache, batchSize * kvLen * dModel);
        oGm.SetGlobalBuffer((__gm__ float*)output, batchSize * qLen * dModel);

        // We process tiles of dModel dimension for dot products
        // and tiles of kvLen for softmax
        // Tile size for dModel dimension
        this->dTileSize = 256;
        if (this->dTileSize > dModel) this->dTileSize = dModel;
        
        // Tile size for kvLen dimension  
        this->kvTileSize = 256;
        if (this->kvTileSize > kvLen) this->kvTileSize = kvLen;

        // Buffers: we need space for q tile, k tile, score accumulation, softmax, v tile, output accumulation
        // q tile: dTileSize floats
        // k tile: dTileSize floats  
        // scores: kvLen floats (aligned)
        // v tile: dTileSize floats
        // out: dModel floats

        uint32_t kvLenAligned = ((kvLen + 63) / 64) * 64;
        uint32_t dModelAligned = ((dModel + 63) / 64) * 64;
        uint32_t dTileSizeAligned = ((dTileSize + 63) / 64) * 64;

        pipe.InitBuffer(qBuf, BUFFER_NUM, dTileSizeAligned * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, dTileSizeAligned * sizeof(float));
        pipe.InitBuffer(scoreBuf, BUFFER_NUM, kvLenAligned * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, dTileSizeAligned * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(tempBuf, BUFFER_NUM, kvLenAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t qi = startQuery; qi < endQuery; qi++) {
            ProcessOneQuery(qi);
        }
    }

private:
    __aicore__ inline void ProcessOneQuery(uint32_t queryIdx)
    {
        uint32_t b = queryIdx / qLen;
        uint32_t q = queryIdx % qLen;

        uint32_t qOffset = b * qLen * dModel + q * dModel;
        uint32_t kBaseOffset = b * kvLen * dModel;
        uint32_t vBaseOffset = b * kvLen * dModel;
        uint32_t oOffset = b * qLen * dModel + q * dModel;

        uint32_t kvLenAligned = ((kvLen + 63) / 64) * 64;
        uint32_t dModelAligned = ((dModel + 63) / 64) * 64;

        // Step 1: Compute attention scores = Q * K^T for all kv positions
        LocalTensor<float> scoreLocal = scoreBuf.AllocTensor<float>();
        // Initialize scores to 0
        Duplicate(scoreLocal, (float)0.0f, kvLenAligned);

        // For each kv position, compute dot product of q and k
        // Process in tiles over dModel dimension
        for (uint32_t kvStart = 0; kvStart < kvLen; kvStart += kvTileSize) {
            uint32_t kvEnd = kvStart + kvTileSize;
            if (kvEnd > kvLen) kvEnd = kvLen;
            uint32_t curKvTile = kvEnd - kvStart;

            // For each kv in this tile
            for (uint32_t kvi = kvStart; kvi < kvEnd; kvi++) {
                float dotProduct = 0.0f;
                uint32_t kOffset = kBaseOffset + kvi * dModel;

                // Compute dot product in tiles over dModel
                for (uint32_t dStart = 0; dStart < dModel; dStart += dTileSize) {
                    uint32_t dEnd = dStart + dTileSize;
                    if (dEnd > dModel) dEnd = dModel;
                    uint32_t curDTile = dEnd - dStart;
                    uint32_t curDTileAligned = ((curDTile + 63) / 64) * 64;

                    LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
                    LocalTensor<float> kLocal = kBuf.AllocTensor<float>();

                    DataCopy(qLocal, qGm[qOffset + dStart], curDTileAligned);
                    DataCopy(kLocal, kGm[kOffset + dStart], curDTileAligned);

                    // Zero out padding if needed
                    if (curDTile < curDTileAligned) {
                        for (uint32_t p = curDTile; p < curDTileAligned; p++) {
                            qLocal.SetValue(p, 0.0f);
                            kLocal.SetValue(p, 0.0f);
                        }
                    }

                    // Element-wise multiply
                    Mul(qLocal, qLocal, kLocal, curDTileAligned);

                    // Reduce sum
                    float partialSum = 0.0f;
                    for (uint32_t idx = 0; idx < curDTile; idx++) {
                        partialSum += qLocal.GetValue(idx);
                    }
                    dotProduct += partialSum;

                    qBuf.FreeTensor(qLocal);
                    kBuf.FreeTensor(kLocal);
                }
                scoreLocal.SetValue(kvi, dotProduct * scale);
            }
        }

        // Step 2: Softmax over scores
        // Find max
        float maxVal = scoreLocal.GetValue(0);
        for (uint32_t i = 1; i < kvLen; i++) {
            float v = scoreLocal.GetValue(i);
            if (v > maxVal) maxVal = v;
        }

        // Subtract max and exp
        float sumExp = 0.0f;
        for (uint32_t i = 0; i < kvLen; i++) {
            float v = scoreLocal.GetValue(i);
            // Simple exp approximation or use built-in
            v = v - maxVal;
            // Use exp
            float ev;
            // Manual exp since we're doing scalar
            // Use a rough exp: we'll store back and use vector Exp
            scoreLocal.SetValue(i, v);
        }

        // Use vector Exp
        LocalTensor<float> tempLocal = tempBuf.AllocTensor<float>();
        Exp(tempLocal, scoreLocal, kvLenAligned);

        // Sum
        sumExp = 0.0f;
        for (uint32_t i = 0; i < kvLen; i++) {
            sumExp += tempLocal.GetValue(i);
        }

        // Normalize
        float invSum = 1.0f / sumExp;
        Muls(scoreLocal, tempLocal, invSum, kvLenAligned);
        tempBuf.FreeTensor(tempLocal);

        // Step 3: Compute output = scores * V
        // output[d] = sum_over_kv(score[kv] * V[kv][d])
        LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
        Duplicate(outLocal, (float)0.0f, dModelAligned);

        for (uint32_t kvi = 0; kvi < kvLen; kvi++) {
            float w = scoreLocal.GetValue(kvi);
            if (w == 0.0f) continue;

            uint32_t vOffset = vBaseOffset + kvi * dModel;

            for (uint32_t dStart = 0; dStart < dModel; dStart += dTileSize) {
                uint32_t dEnd = dStart + dTileSize;
                if (dEnd > dModel) dEnd = dModel;
                uint32_t curDTile = dEnd - dStart;
                uint32_t curDTileAligned = ((curDTile + 63) / 64) * 64;

                LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
                DataCopy(vLocal, vGm[vOffset + dStart], curDTileAligned);

                // Multiply by weight
                Muls(vLocal, vLocal, w, curDTileAligned);

                // Add to output - we need to read outLocal segment, add, write back
                // Since outLocal is already allocated, we can do element-wise
                for (uint32_t idx = 0; idx < curDTile; idx++) {
                    float cur = outLocal.GetValue(dStart + idx);
                    cur += vLocal.GetValue(idx);
                    outLocal.SetValue(dStart + idx, cur);
                }

                vBuf.FreeTensor(vLocal);
            }
        }

        scoreBuf.FreeTensor(scoreLocal);

        // Copy output to global memory
        DataCopy(oGm[oOffset], outLocal, dModelAligned);
        outBuf.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> qBuf, kBuf, vBuf;
    TQue<TPosition::VECOUT, BUFFER_NUM> scoreBuf, outBuf, tempBuf;
    GlobalTensor<float> qGm, kGm, vGm, oGm;
    uint32_t batchSize, qLen, kvLen, dModel;
    uint32_t dTileSize, kvTileSize;
    uint32_t startQuery, endQuery;
    float scale;
};

extern "C" __global__ __aicore__ void kv_cached_speculative_attention_custom(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAttention op;
    op.Init(query, key_cache, value_cache, output,
            tiling_data.batchSize, tiling_data.qLen, tiling_data.kvLen, tiling_data.dModel, tiling_data.scale);
    op.Process();
}
