
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGroupQueryAttention {
public:
    __aicore__ inline KernelGroupQueryAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                 uint32_t batchSize, uint32_t seqLen,
                                 uint32_t numHeads, uint32_t numKvHeads,
                                 uint32_t headDim, uint32_t groupSize)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->numHeads = numHeads;
        this->numKvHeads = numKvHeads;
        this->headDim = headDim;
        this->groupSize = groupSize;

        uint32_t totalBH = batchSize * numHeads;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->taskStart = blockIdx * ((totalBH + numBlocks - 1) / numBlocks);
        this->taskEnd = (blockIdx + 1) * ((totalBH + numBlocks - 1) / numBlocks);
        if (this->taskEnd > totalBH) this->taskEnd = totalBH;

        // Q layout: [B, L, H, headDim], K layout: [B, L, H_kv, headDim], V same as K
        qGm.SetGlobalBuffer((__gm__ float*)q, batchSize * seqLen * numHeads * headDim);
        kGm.SetGlobalBuffer((__gm__ float*)k, batchSize * seqLen * numKvHeads * headDim);
        vGm.SetGlobalBuffer((__gm__ float*)v, batchSize * seqLen * numKvHeads * headDim);
        outGm.SetGlobalBuffer((__gm__ float*)out, batchSize * seqLen * numHeads * headDim);

        // Allocate buffers for tiles
        // We process one query row at a time against all key rows
        // qRow: [headDim], kTile: [seqLen * headDim], scores: [seqLen], vTile: [seqLen * headDim]
        // For small seqLen (128) and headDim (64), this fits in UB
        uint32_t alignedHeadDim = ((headDim + 7) / 8) * 8;
        uint32_t alignedSeqLen = ((seqLen + 7) / 8) * 8;

        this->alignedHeadDim = alignedHeadDim;
        this->alignedSeqLen = alignedSeqLen;

        pipe.InitBuffer(qBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(scoresBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
        pipe.InitBuffer(maxBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = this->taskStart; task < this->taskEnd; task++) {
            uint32_t b = task / numHeads;
            uint32_t h = task % numHeads;
            uint32_t kv_h = h / groupSize;
            ProcessOneHead(b, h, kv_h);
        }
    }

private:
    __aicore__ inline void ProcessOneHead(uint32_t b, uint32_t h, uint32_t kv_h)
    {
        float scale = 1.0f;
        // Compute 1/sqrt(headDim)
        for (uint32_t i = 0; i < headDim; i++) {
            scale = 1.0f;
        }
        // Manual inverse sqrt
        float invSqrt = 1.0f;
        {
            float hd = (float)headDim;
            // Newton's method for 1/sqrt(hd)
            // Initial guess
            invSqrt = 1.0f;
            if (headDim == 64) invSqrt = 0.125f;
            else if (headDim == 128) invSqrt = 0.0883883f;
            else if (headDim == 32) invSqrt = 0.176777f;
            else {
                // rough approximation
                invSqrt = 1.0f;
                for (int iter = 0; iter < 10; iter++) {
                    invSqrt = invSqrt * (1.5f - 0.5f * hd * invSqrt * invSqrt);
                }
            }
        }

        for (uint32_t ql = 0; ql < seqLen; ql++) {
            // Load Q[b, ql, h, :] 
            AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> scoresLocal = scoresBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> maxLocal = maxBuf.AllocTensor<float>();

            // Q offset: b * seqLen * numHeads * headDim + ql * numHeads * headDim + h * headDim
            uint32_t qOff = b * seqLen * numHeads * headDim + ql * numHeads * headDim + h * headDim;
            AscendC::DataCopy(qLocal, qGm[qOff], alignedHeadDim);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Compute attention scores for all keys
            // scores[kl] = sum_d Q[d] * K[b, kl, kv_h, d] / sqrt(headDim)
            // We compute this manually element by element for correctness
            for (uint32_t kl = 0; kl < seqLen; kl++) {
                uint32_t kOff = b * seqLen * numKvHeads * headDim + kl * numKvHeads * headDim + kv_h * headDim;
                
                // Load K row into tmpLocal (reusing)
                AscendC::DataCopy(outLocal, kGm[kOff], alignedHeadDim);
                AscendC::PipeBarrier<PIPE_ALL>();
                
                // Dot product: qLocal * outLocal
                AscendC::Mul(tmpLocal, qLocal, outLocal, alignedHeadDim);
                AscendC::PipeBarrier<PIPE_ALL>();
                
                // Sum reduction
                float dotVal = 0.0f;
                for (uint32_t d = 0; d < headDim; d++) {
                    dotVal += tmpLocal.GetValue(d);
                }
                scoresLocal.SetValue(kl, dotVal * invSqrt);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Softmax over scores
            // Find max
            float maxVal = scoresLocal.GetValue(0);
            for (uint32_t kl = 1; kl < seqLen; kl++) {
                float val = scoresLocal.GetValue(kl);
                if (val > maxVal) maxVal = val;
            }
            // Exp and sum
            float sumExp = 0.0f;
            for (uint32_t kl = 0; kl < seqLen; kl++) {
                float val = scoresLocal.GetValue(kl) - maxVal;
                // Approximate exp
                float expVal;
                // Use built-in if possible, otherwise Taylor
                // exp(x) approximation
                expVal = 1.0f;
                float term = 1.0f;
                for (int t = 1; t <= 12; t++) {
                    term *= val / (float)t;
                    expVal += term;
                }
                if (expVal < 0.0f) expVal = 0.0f;
                scoresLocal.SetValue(kl, expVal);
                sumExp += expVal;
            }
            // Normalize
            float invSum = 1.0f / sumExp;
            for (uint32_t kl = 0; kl < seqLen; kl++) {
                scoresLocal.SetValue(kl, scoresLocal.GetValue(kl) * invSum);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Compute weighted sum of V
            // out[d] = sum_kl scores[kl] * V[b, kl, kv_h, d]
            for (uint32_t d = 0; d < headDim; d++) {
                outLocal.SetValue(d, 0.0f);
            }
            for (uint32_t kl = 0; kl < seqLen; kl++) {
                float w = scoresLocal.GetValue(kl);
                uint32_t vOff = b * seqLen * numKvHeads * headDim + kl * numKvHeads * headDim + kv_h * headDim;
                AscendC::DataCopy(tmpLocal, vGm[vOff], alignedHeadDim);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t d = 0; d < headDim; d++) {
                    outLocal.SetValue(d, outLocal.GetValue(d) + w * tmpLocal.GetValue(d));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Write output: out[b, ql, h, :]
            uint32_t outOff = b * seqLen * numHeads * headDim + ql * numHeads * headDim + h * headDim;
            AscendC::DataCopy(outGm[outOff], outLocal, alignedHeadDim);
            AscendC::PipeBarrier<PIPE_ALL>();

            qBuf.FreeTensor(qLocal);
            scoresBuf.FreeTensor(scoresLocal);
            outBuf.FreeTensor(outLocal);
            tmpBuf.FreeTensor(tmpLocal);
            maxBuf.FreeTensor(maxLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qBuf, scoresBuf, tmpBuf, maxBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm;
    uint32_t batchSize, seqLen, numHeads, numKvHeads, headDim, groupSize;
    uint32_t alignedHeadDim, alignedSeqLen;
    uint32_t taskStart, taskEnd;
};

extern "C" __global__ __aicore__ void group_query_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupQueryAttention op;
    op.Init(q, k, v, out,
            tiling_data.batchSize, tiling_data.seqLen,
            tiling_data.numHeads, tiling_data.numKvHeads,
            tiling_data.headDim, tiling_data.groupSize);
    op.Process();
}
