
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCrossModalAttention {
public:
    __aicore__ inline KernelCrossModalAttention() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLenQ, uint32_t seqLenKV,
                                 uint32_t headDim, float scale)
    {
        this->batchSize = batchSize;
        this->seqLenQ = seqLenQ;
        this->seqLenKV = seqLenKV;
        this->headDim = headDim;
        this->scale = scale;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();

        // Number of (batch, head) items each block processes
        this->itemsPerBlock = (batchSize + numBlocks - 1) / numBlocks;
        this->startItem = blockIdx * this->itemsPerBlock;
        this->endItem = startItem + itemsPerBlock;
        if (this->endItem > batchSize) this->endItem = batchSize;

        uint32_t qStride = seqLenQ * headDim;
        uint32_t kvStride = seqLenKV * headDim;

        qGm.SetGlobalBuffer((__gm__ float*)q, batchSize * qStride);
        kGm.SetGlobalBuffer((__gm__ float*)k, batchSize * kvStride);
        vGm.SetGlobalBuffer((__gm__ float*)v, batchSize * kvStride);
        outGm.SetGlobalBuffer((__gm__ float*)out, batchSize * qStride);
        wsGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize * seqLenQ * seqLenKV);

        // Align headDim and seqLenKV to 8 for float (32 bytes)
        uint32_t headDimAligned = ((headDim + 7) / 8) * 8;
        uint32_t seqLenKVAligned = ((seqLenKV + 7) / 8) * 8;

        // Buffer sizes
        uint32_t qRowSize = headDimAligned * sizeof(float);
        uint32_t kColSize = headDimAligned * sizeof(float);
        uint32_t scoreRowSize = seqLenKVAligned * sizeof(float);
        uint32_t vRowSize = headDimAligned * sizeof(float);

        pipe.InitBuffer(qBuf, BUFFER_NUM, qRowSize);
        pipe.InitBuffer(kBuf, BUFFER_NUM, kColSize);
        pipe.InitBuffer(scoreBuf, BUFFER_NUM, scoreRowSize);
        pipe.InitBuffer(softmaxBuf, BUFFER_NUM, scoreRowSize);
        pipe.InitBuffer(vBufLocal, BUFFER_NUM, vRowSize);
        pipe.InitBuffer(outBuf, BUFFER_NUM, headDimAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, headDimAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t qStride = seqLenQ * headDim;
        uint32_t kvStride = seqLenKV * headDim;

        for (uint32_t item = startItem; item < endItem; item++) {
            uint32_t qBase = item * qStride;
            uint32_t kvBase = item * kvStride;
            uint32_t outBase = item * qStride;
            uint32_t wsBase = item * seqLenQ * seqLenKV;

            for (uint32_t qi = 0; qi < seqLenQ; qi++) {
                // Load Q row
                AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
                AscendC::DataCopy(qLocal, qGm[qBase + qi * headDim], headDim);
                AscendC::PipeBarrier<PIPE_ALL>();

                // Compute attention scores for this query row against all keys
                AscendC::LocalTensor<float> scoreLocal = scoreBuf.AllocTensor<float>();

                for (uint32_t ki = 0; ki < seqLenKV; ki++) {
                    // Load K row
                    AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
                    AscendC::DataCopy(kLocal, kGm[kvBase + ki * headDim], headDim);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // Dot product: q * k
                    AscendC::LocalTensor<float> tLocal = tmpBuf.AllocTensor<float>();
                    AscendC::Mul(tLocal, qLocal, kLocal, headDim);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // Sum reduction
                    float dotVal = 0.0f;
                    uint32_t hdAligned = ((headDim + 7) / 8) * 8;
                    // Use ReduceSum if available, else manual
                    float sum = 0.0f;
                    for (uint32_t d = 0; d < headDim; d++) {
                        sum += tLocal.GetValue(d);
                    }
                    scoreLocal.SetValue(ki, sum * scale);

                    tmpBuf.FreeTensor(tLocal);
                    kBuf.FreeTensor(kLocal);
                }

                AscendC::PipeBarrier<PIPE_ALL>();

                // Softmax over seqLenKV scores
                // Find max
                AscendC::LocalTensor<float> smLocal = softmaxBuf.AllocTensor<float>();
                float maxVal = scoreLocal.GetValue(0);
                for (uint32_t j = 1; j < seqLenKV; j++) {
                    float sv = scoreLocal.GetValue(j);
                    if (sv > maxVal) maxVal = sv;
                }
                // exp and sum
                float expSum = 0.0f;
                for (uint32_t j = 0; j < seqLenKV; j++) {
                    float ev = scoreLocal.GetValue(j) - maxVal;
                    // Approximate exp or use scalar
                    float expVal = 1.0f;
                    // Simple exp computation via Taylor or built-in
                    // Using scalar exp approximation
                    float x = ev;
                    // Use a reasonable exp approximation
                    // exp(x) for x <= 0
                    if (x < -10.0f) {
                        expVal = 0.0f;
                    } else {
                        expVal = 1.0f + x;
                        float term = x;
                        term *= x / 2.0f; expVal += term;
                        term *= x / 3.0f; expVal += term;
                        term *= x / 4.0f; expVal += term;
                        term *= x / 5.0f; expVal += term;
                        term *= x / 6.0f; expVal += term;
                        term *= x / 7.0f; expVal += term;
                        term *= x / 8.0f; expVal += term;
                        if (expVal < 0.0f) expVal = 0.0f;
                    }
                    smLocal.SetValue(j, expVal);
                    expSum += expVal;
                }
                // Normalize
                float invSum = (expSum > 0.0f) ? (1.0f / expSum) : 0.0f;
                for (uint32_t j = 0; j < seqLenKV; j++) {
                    smLocal.SetValue(j, smLocal.GetValue(j) * invSum);
                }

                AscendC::PipeBarrier<PIPE_ALL>();

                // Compute output: weighted sum of V rows
                AscendC::LocalTensor<float> oLocal = outBuf.AllocTensor<float>();
                for (uint32_t d = 0; d < headDim; d++) {
                    oLocal.SetValue(d, 0.0f);
                }

                for (uint32_t vi = 0; vi < seqLenKV; vi++) {
                    float w = smLocal.GetValue(vi);
                    if (w > 0.0f) {
                        AscendC::LocalTensor<float> vLocal = vBufLocal.AllocTensor<float>();
                        AscendC::DataCopy(vLocal, vGm[kvBase + vi * headDim], headDim);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t d = 0; d < headDim; d++) {
                            oLocal.SetValue(d, oLocal.GetValue(d) + w * vLocal.GetValue(d));
                        }
                        vBufLocal.FreeTensor(vLocal);
                    }
                }

                AscendC::PipeBarrier<PIPE_ALL>();

                // Write output
                AscendC::DataCopy(outGm[outBase + qi * headDim], oLocal, headDim);
                AscendC::PipeBarrier<PIPE_ALL>();

                outBuf.FreeTensor(oLocal);
                softmaxBuf.FreeTensor(smLocal);
                scoreBuf.FreeTensor(scoreLocal);
                qBuf.FreeTensor(qLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qBuf, kBuf, vBufLocal;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outBuf;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> scoreBuf, softmaxBuf, tmpBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm, wsGm;
    uint32_t batchSize, seqLenQ, seqLenKV, headDim;
    uint32_t itemsPerBlock, startItem, endItem;
    float scale;
};

extern "C" __global__ __aicore__ void cross_modal_attention_custom(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossModalAttention op;
    op.Init(q, k, v, out, workspace,
            tiling_data.batchSize, tiling_data.seqLenQ, tiling_data.seqLenKV,
            tiling_data.headDim, tiling_data.scale);
    op.Process();
}
