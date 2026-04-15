
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMultiQueryAttention {
public:
    __aicore__ inline KernelMultiQueryAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t numHeads, uint32_t headDim)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->numHeads = numHeads;
        this->headDim = headDim;

        // blockIdx maps to (batch, head)
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t bIdx = blockIdx / numHeads;
        uint32_t hIdx = blockIdx % numHeads;

        this->bIdx = bIdx;
        this->hIdx = hIdx;

        // Q shape: [B, N, H, d_h] stored as [B, N, H*d_h]
        // Q for this (b, h): stride b -> N*H*d_h, for each n -> offset h*d_h
        // K shape: [B, N, 1, d_h] stored as [B, N, d_h]
        // V shape: [B, N, 1, d_h] stored as [B, N, d_h]
        // Out shape: [B, N, H*d_h]

        uint32_t qBaseOffset = bIdx * seqLen * numHeads * headDim;
        uint32_t kvBaseOffset = bIdx * seqLen * headDim;
        uint32_t outBaseOffset = bIdx * seqLen * numHeads * headDim;

        qGm.SetGlobalBuffer((__gm__ float*)q + qBaseOffset, seqLen * numHeads * headDim);
        kGm.SetGlobalBuffer((__gm__ float*)k + kvBaseOffset, seqLen * headDim);
        vGm.SetGlobalBuffer((__gm__ float*)v + kvBaseOffset, seqLen * headDim);
        outGm.SetGlobalBuffer((__gm__ float*)out + outBaseOffset, seqLen * numHeads * headDim);

        // workspace for attn scores: seqLen * seqLen per block
        uint32_t wsOffset = blockIdx * seqLen * seqLen;
        wsGm.SetGlobalBuffer((__gm__ float*)workspace + wsOffset, seqLen * seqLen);

        // Allocate buffers
        // We process tile by tile along the query sequence dimension
        // For each query position, compute dot products with all keys, softmax, then weighted sum of values
        uint32_t alignedHeadDim = ((headDim + 7) / 8) * 8;
        uint32_t alignedSeqLen = ((seqLen + 7) / 8) * 8;

        this->alignedHeadDim = alignedHeadDim;
        this->alignedSeqLen = alignedSeqLen;

        pipe.InitBuffer(qBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(scoreBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // For each query position n:
        //   1. Load Q[b, n, h, :] (headDim)
        //   2. For each key position k: compute dot(Q, K[b, k, :]) / sqrt(d_h)
        //   3. Softmax over k dimension
        //   4. Weighted sum of V
        //   5. Store to out[b, n, h, :]

        float invSqrt = 1.0f / sqrtf((float)headDim);

        for (uint32_t n = 0; n < seqLen; n++) {
            // Load Q[b, n, h, :] - Q is [B, N, H*d_h], so offset = n*H*d_h + h*d_h
            AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
            uint32_t qOffset = n * numHeads * headDim + hIdx * headDim;

            // Use DataCopy for aligned headDim
            if (headDim == alignedHeadDim) {
                AscendC::DataCopy(qLocal, qGm[qOffset], headDim);
            } else {
                AscendC::DataCopy(qLocal, qGm[qOffset], alignedHeadDim);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Compute attention scores
            AscendC::LocalTensor<float> scoreLocal = scoreBuf.AllocTensor<float>();

            // Compute dot products with all keys
            for (uint32_t ki = 0; ki < seqLen; ki++) {
                AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
                uint32_t kOffset = ki * headDim;
                if (headDim == alignedHeadDim) {
                    AscendC::DataCopy(kLocal, kGm[kOffset], headDim);
                } else {
                    AscendC::DataCopy(kLocal, kGm[kOffset], alignedHeadDim);
                }
                AscendC::PipeBarrier<PIPE_ALL>();

                // Element-wise multiply and reduce
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                AscendC::Mul(tmpLocal, qLocal, kLocal, alignedHeadDim);
                AscendC::PipeBarrier<PIPE_ALL>();

                // Sum reduction
                float dotVal = 0.0f;
                for (uint32_t d = 0; d < headDim; d++) {
                    dotVal += tmpLocal.GetValue(d);
                }
                scoreLocal.SetValue(ki, dotVal * invSqrt);

                tmpBuf.FreeTensor(tmpLocal);
                kBuf.FreeTensor(kLocal);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Softmax: find max, subtract, exp, sum, divide
            float maxVal = scoreLocal.GetValue(0);
            for (uint32_t ki = 1; ki < seqLen; ki++) {
                float sv = scoreLocal.GetValue(ki);
                if (sv > maxVal) maxVal = sv;
            }
            float sumExp = 0.0f;
            for (uint32_t ki = 0; ki < seqLen; ki++) {
                float ev = expf(scoreLocal.GetValue(ki) - maxVal);
                scoreLocal.SetValue(ki, ev);
                sumExp += ev;
            }
            float invSum = 1.0f / sumExp;
            for (uint32_t ki = 0; ki < seqLen; ki++) {
                scoreLocal.SetValue(ki, scoreLocal.GetValue(ki) * invSum);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Weighted sum of values
            AscendC::LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
            // Initialize output to zero
            for (uint32_t d = 0; d < alignedHeadDim; d++) {
                outLocal.SetValue(d, 0.0f);
            }

            for (uint32_t ki = 0; ki < seqLen; ki++) {
                float weight = scoreLocal.GetValue(ki);
                AscendC::LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
                uint32_t vOffset = ki * headDim;
                if (headDim == alignedHeadDim) {
                    AscendC::DataCopy(vLocal, vGm[vOffset], headDim);
                } else {
                    AscendC::DataCopy(vLocal, vGm[vOffset], alignedHeadDim);
                }
                AscendC::PipeBarrier<PIPE_ALL>();

                for (uint32_t d = 0; d < headDim; d++) {
                    outLocal.SetValue(d, outLocal.GetValue(d) + weight * vLocal.GetValue(d));
                }
                vBuf.FreeTensor(vLocal);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Store output: out[b, n, h*d_h + d] for d in [0, headDim)
            uint32_t outOffset = n * numHeads * headDim + hIdx * headDim;
            if (headDim == alignedHeadDim) {
                AscendC::DataCopy(outGm[outOffset], outLocal, headDim);
            } else {
                AscendC::DataCopy(outGm[outOffset], outLocal, alignedHeadDim);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            outBuf.FreeTensor(outLocal);
            scoreBuf.FreeTensor(scoreLocal);
            qBuf.FreeTensor(qLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf, kBuf, vBuf, scoreBuf, outBuf, tmpBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm, wsGm;
    uint32_t batchSize, seqLen, numHeads, headDim;
    uint32_t bIdx, hIdx;
    uint32_t alignedHeadDim, alignedSeqLen;
};

extern "C" __global__ __aicore__ void multi_query_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiQueryAttention op;
    op.Init(q, k, v, out, workspace, tiling_data.batchSize, tiling_data.seqLen, tiling_data.numHeads, tiling_data.headDim);
    op.Process();
}
