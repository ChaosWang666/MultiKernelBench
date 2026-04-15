
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelWindowedCausalAttention {
public:
    __aicore__ inline KernelWindowedCausalAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t numHeads,
                                 uint32_t headDim, uint32_t windowSize)
    {
        this->B = batchSize;
        this->N = seqLen;
        this->H = numHeads;
        this->d_h = headDim;
        this->W = windowSize;

        // Each block handles one (b, h) pair
        uint32_t blockIdx = GetBlockIdx();
        uint32_t b = blockIdx / H;
        uint32_t h = blockIdx % H;

        // Q, K, V layout: [B, N, H, d_h] -> stride: N*H*d_h, H*d_h, d_h, 1
        uint32_t bhOffset = b * N * H * d_h + h * d_h;

        qGm.SetGlobalBuffer((__gm__ float*)q + bhOffset, N * H * d_h);
        kGm.SetGlobalBuffer((__gm__ float*)k + bhOffset, N * H * d_h);
        vGm.SetGlobalBuffer((__gm__ float*)v + bhOffset, N * H * d_h);
        outGm.SetGlobalBuffer((__gm__ float*)out + bhOffset, N * H * d_h);

        // Align headDim to 8 for float (32 bytes)
        uint32_t alignedDh = ((d_h + 7) / 8) * 8;
        uint32_t maxWin = W + 1;
        uint32_t alignedMaxWin = ((maxWin + 7) / 8) * 8;

        // Buffer sizes
        // qBuf: alignedDh floats for one query
        // kBuf: alignedDh floats for one key
        // vBuf: alignedDh floats for one value
        // scoreBuf: alignedMaxWin floats for scores
        // outBuf: alignedDh floats for output accumulation
        // tmpBuf: alignedDh floats for temporary multiply results

        pipe.InitBuffer(qBufQueue, 1, alignedDh * sizeof(float));
        pipe.InitBuffer(outBufQueue, 1, alignedDh * sizeof(float));
        pipe.InitBuffer(scoreBufQueue, 1, alignedMaxWin * sizeof(float));
        pipe.InitBuffer(tmpBufQueue, 1, alignedDh * sizeof(float));
        pipe.InitBuffer(kBufQueue, 1, alignedDh * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t stride = H * d_h;
        uint32_t alignedDh = ((d_h + 7) / 8) * 8;
        uint32_t maxWin = W + 1;
        uint32_t alignedMaxWin = ((maxWin + 7) / 8) * 8;

        for (uint32_t i = 0; i < N; i++) {
            uint32_t start = (i > W) ? (i - W) : 0;
            uint32_t winLen = i - start + 1;

            // Load query for position i
            LocalTensor<float> qLocal = qBufQueue.AllocTensor<float>();
            DataCopy(qLocal, qGm[i * stride], alignedDh);
            pipe_barrier(PIPE_ALL);

            // Compute dot products: score[j] = Q_i . K_{start+j} / sqrt(d_h)
            LocalTensor<float> scoreLocal = scoreBufQueue.AllocTensor<float>();
            // Zero out score buffer
            Duplicate(scoreLocal, (float)0.0f, alignedMaxWin);
            pipe_barrier(PIPE_ALL);

            LocalTensor<float> kLocal = kBufQueue.AllocTensor<float>();
            LocalTensor<float> tmpLocal = tmpBufQueue.AllocTensor<float>();

            for (uint32_t j = 0; j < winLen; j++) {
                uint32_t kIdx = (start + j) * stride;
                DataCopy(kLocal, kGm[kIdx], alignedDh);
                pipe_barrier(PIPE_ALL);

                // Element-wise multiply
                Mul(tmpLocal, qLocal, kLocal, alignedDh);
                pipe_barrier(PIPE_ALL);

                // Sum reduction - accumulate manually
                float dotVal = 0.0f;
                for (uint32_t dd = 0; dd < d_h; dd++) {
                    dotVal += tmpLocal.GetValue(dd);
                }
                // Scale by 1/sqrt(d_h)
                float scale = 1.0f;
                float sqrtDh = 1.0f;
                // Compute sqrt(d_h)
                for (uint32_t iter = 0; iter < 20; iter++) {
                    if (iter == 0) sqrtDh = (float)d_h / 2.0f;
                    sqrtDh = 0.5f * (sqrtDh + (float)d_h / sqrtDh);
                }
                scale = 1.0f / sqrtDh;
                scoreLocal.SetValue(j, dotVal * scale);
            }
            pipe_barrier(PIPE_ALL);

            // Softmax over winLen elements
            // Find max
            float maxVal = scoreLocal.GetValue(0);
            for (uint32_t j = 1; j < winLen; j++) {
                float sv = scoreLocal.GetValue(j);
                if (sv > maxVal) maxVal = sv;
            }

            // Exp and sum
            float sumExp = 0.0f;
            for (uint32_t j = 0; j < winLen; j++) {
                float ev = scoreLocal.GetValue(j) - maxVal;
                // Compute exp approximation or use built-in
                // Use simple exp via setting value
                float expVal;
                // Manual exp using Taylor or platform exp
                // For correctness, we use a loop-based approach
                expVal = 1.0f;
                float term = 1.0f;
                for (int t = 1; t <= 12; t++) {
                    term *= ev / (float)t;
                    expVal += term;
                }
                if (expVal < 0.0f) expVal = 0.0f;
                scoreLocal.SetValue(j, expVal);
                sumExp += expVal;
            }

            // Normalize
            float invSum = 1.0f / sumExp;
            for (uint32_t j = 0; j < winLen; j++) {
                scoreLocal.SetValue(j, scoreLocal.GetValue(j) * invSum);
            }
            pipe_barrier(PIPE_ALL);

            // Weighted sum of values
            LocalTensor<float> outLocal = outBufQueue.AllocTensor<float>();
            Duplicate(outLocal, (float)0.0f, alignedDh);
            pipe_barrier(PIPE_ALL);

            for (uint32_t j = 0; j < winLen; j++) {
                uint32_t vIdx = (start + j) * stride;
                // Reuse kLocal buffer for loading V
                DataCopy(kLocal, vGm[vIdx], alignedDh);
                pipe_barrier(PIPE_ALL);

                float w = scoreLocal.GetValue(j);
                // tmpLocal = w * V[j]
                Muls(tmpLocal, kLocal, w, alignedDh);
                pipe_barrier(PIPE_ALL);

                // outLocal += tmpLocal
                Add(outLocal, outLocal, tmpLocal, alignedDh);
                pipe_barrier(PIPE_ALL);
            }

            // Write output
            DataCopy(outGm[i * stride], outLocal, alignedDh);
            pipe_barrier(PIPE_ALL);

            kBufQueue.FreeTensor(kLocal);
            tmpBufQueue.FreeTensor(tmpLocal);
            scoreBufQueue.FreeTensor(scoreLocal);
            outBufQueue.FreeTensor(outLocal);
            qBufQueue.FreeTensor(qLocal);
        }
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> qBufQueue, kBufQueue;
    TQue<TPosition::VECOUT, 1> outBufQueue;
    TQue<TPosition::VECIN, 1> scoreBufQueue, tmpBufQueue;
    GlobalTensor<float> qGm, kGm, vGm, outGm;
    uint32_t B, N, H, d_h, W;
};

extern "C" __global__ __aicore__ void windowed_causal_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelWindowedCausalAttention op;
    op.Init(q, k, v, out, workspace,
            tiling_data.batchSize, tiling_data.seqLen,
            tiling_data.numHeads, tiling_data.headDim, tiling_data.windowSize);
    op.Process();
}
