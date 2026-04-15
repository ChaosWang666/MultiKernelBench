
#include "kernel_operator.h"

// KV-cached attention inference kernel
// Q: (batch_size, q_len, d_model), K: (batch_size, kv_len, d_model), V: (batch_size, kv_len, d_model)
// For inference: batch_size=1, q_len=1
// Output: (batch_size, q_len, d_model)
// Computes: softmax(Q * K^T / sqrt(d_model)) * V

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelKvCachedAttention {
public:
    __aicore__ inline KernelKvCachedAttention() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR output,
                                 GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t qLen, uint32_t kvLen, uint32_t dModel, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->qLen = qLen;
        this->kvLen = kvLen;
        this->dModel = dModel;
        this->tileNum = tileNum;

        // Global memory pointers
        qGm.SetGlobalBuffer((__gm__ float *)query, batchSize * qLen * dModel);
        kGm.SetGlobalBuffer((__gm__ float *)key_cache, batchSize * kvLen * dModel);
        vGm.SetGlobalBuffer((__gm__ float *)value_cache, batchSize * kvLen * dModel);
        outGm.SetGlobalBuffer((__gm__ float *)output, batchSize * qLen * dModel);

        // Use workspace for scores
        scoresGm.SetGlobalBuffer((__gm__ float *)workspace, batchSize * qLen * kvLen);

        // Align kvLen and dModel to 8 for data copy
        uint32_t kvLenAligned = ((kvLen + 7) / 8) * 8;
        uint32_t dModelAligned = ((dModel + 7) / 8) * 8;

        // Buffer allocation:
        // We need buffers for:
        // - q tile (part of dModel)
        // - k tile (part of kv_len x part of dModel)
        // - scores (kvLen)
        // - v tile
        // - output tile

        // We'll process d_model in tiles
        this->dTileSize = dModel / tileNum;
        if (this->dTileSize == 0) this->dTileSize = dModel;
        uint32_t dTileAligned = ((this->dTileSize + 7) / 8) * 8;

        pipe.InitBuffer(inQueueQ, BUFFER_NUM, dTileAligned * sizeof(float));
        pipe.InitBuffer(inQueueK, BUFFER_NUM, dTileAligned * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(outQueueScore, BUFFER_NUM, kvLenAligned * sizeof(float));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, kvLenAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // For each batch and each query position
        for (uint32_t b = 0; b < batchSize; b++) {
            for (uint32_t q = 0; q < qLen; q++) {
                ComputeAttention(b, q);
            }
        }
    }

private:
    __aicore__ inline void ComputeAttention(uint32_t b, uint32_t q)
    {
        uint32_t kvLenAligned = ((kvLen + 7) / 8) * 8;
        uint32_t dModelAligned = ((dModel + 7) / 8) * 8;
        uint32_t dTileAligned = ((dTileSize + 7) / 8) * 8;

        float scale = 1.0f / sqrtf((float)dModel);

        // Step 1: Compute scores = Q * K^T, scores shape: (kvLen,)
        // Initialize scores to 0
        LocalTensor<float> scoresLocal = outQueueScore.AllocTensor<float>();
        Duplicate(scoresLocal, 0.0f, kvLenAligned);

        uint32_t qOffset = b * qLen * dModel + q * dModel;

        for (uint32_t kv = 0; kv < kvLen; kv++) {
            uint32_t kOffset = b * kvLen * dModel + kv * dModel;
            float dotVal = 0.0f;

            // Compute dot product in tiles along dModel dimension
            for (uint32_t dt = 0; dt < dModel; dt += dTileSize) {
                uint32_t currentTile = dTileSize;
                if (dt + currentTile > dModel) currentTile = dModel - dt;
                uint32_t currentTileAligned = ((currentTile + 7) / 8) * 8;

                LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
                LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();

                // Copy Q tile
                DataCopy(qLocal, qGm[qOffset + dt], currentTileAligned);
                // Copy K tile
                DataCopy(kLocal, kGm[kOffset + dt], currentTileAligned);

                // Multiply element-wise
                Mul(qLocal, qLocal, kLocal, currentTileAligned);

                // Sum reduction - accumulate
                // We reduce by summing all elements
                float tileSum = 0.0f;
                // Use ReduceSum if available, otherwise manual
                LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                ReduceSum(tmpLocal, qLocal, tmpBuf, currentTileAligned);
                tileSum = tmpLocal.GetValue(0);
                tmpBuf.FreeTensor(tmpLocal);

                dotVal += tileSum;

                inQueueQ.FreeTensor(qLocal);
                inQueueK.FreeTensor(kLocal);
            }

            scoresLocal.SetValue(kv, dotVal * scale);
        }

        // Step 2: Softmax over scores
        // Find max
        LocalTensor<float> tmpLocal2 = tmpBuf.AllocTensor<float>();
        ReduceMax(tmpLocal2, scoresLocal, tmpBuf, kvLenAligned);
        float maxVal = tmpLocal2.GetValue(0);
        tmpBuf.FreeTensor(tmpLocal2);

        // Subtract max and exp
        Adds(scoresLocal, scoresLocal, -maxVal, kvLenAligned);
        Exp(scoresLocal, scoresLocal, kvLenAligned);

        // Sum
        LocalTensor<float> tmpLocal3 = tmpBuf.AllocTensor<float>();
        ReduceSum(tmpLocal3, scoresLocal, tmpBuf, kvLenAligned);
        float sumVal = tmpLocal3.GetValue(0);
        tmpBuf.FreeTensor(tmpLocal3);

        // Divide
        float invSum = 1.0f / sumVal;
        Muls(scoresLocal, scoresLocal, invSum, kvLenAligned);

        // Step 3: Compute output = scores * V, output shape: (dModel,)
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        Duplicate(outLocal, 0.0f, dModelAligned);

        for (uint32_t kv = 0; kv < kvLen; kv++) {
            float scoreVal = scoresLocal.GetValue(kv);
            if (scoreVal == 0.0f) continue;

            uint32_t vOffset = b * kvLen * dModel + kv * dModel;
            LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
            DataCopy(vLocal, vGm[vOffset], dModelAligned);

            // outLocal += scoreVal * vLocal
            Muls(vLocal, vLocal, scoreVal, dModelAligned);
            Add(outLocal, outLocal, vLocal, dModelAligned);

            inQueueV.FreeTensor(vLocal);
        }

        // Write output
        uint32_t outOffset = b * qLen * dModel + q * dModel;
        DataCopy(outGm[outOffset], outLocal, dModelAligned);

        outQueueOut.FreeTensor(outLocal);
        outQueueScore.FreeTensor(scoresLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueQ, inQueueK, inQueueV;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueScore, outQueueOut;
    TBuf<TPosition::VECCALC> tmpBuf;
    GlobalTensor<float> qGm, kGm, vGm, outGm, scoresGm;
    uint32_t batchSize, qLen, kvLen, dModel, tileNum, dTileSize;
};

extern "C" __global__ __aicore__ void kv_cached_attention_inference_custom(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelKvCachedAttention op;
    op.Init(query, key_cache, value_cache, output, workspace,
            tiling_data.batchSize, tiling_data.qLen, tiling_data.kvLen, tiling_data.dModel, tiling_data.tileNum);
    op.Process();
}
