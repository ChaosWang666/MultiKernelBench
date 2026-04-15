
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSparseAttention {
public:
    __aicore__ inline KernelSparseAttention() {}
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t numHeads, uint32_t seqLen, uint32_t headDim,
                                 uint32_t windowSize, uint32_t totalBatchHeads)
    {
        this->seqLen = seqLen;
        this->headDim = headDim;
        this->windowSize = windowSize;
        this->totalBatchHeads = totalBatchHeads;

        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->tasksPerBlock = (totalBatchHeads + numBlocks - 1) / numBlocks;
        this->startTask = blockIdx * this->tasksPerBlock;
        this->endTask = startTask + tasksPerBlock;
        if (this->endTask > totalBatchHeads) this->endTask = totalBatchHeads;

        uint32_t headStride = seqLen * headDim;

        qGm.SetGlobalBuffer((__gm__ float*)query, totalBatchHeads * headStride);
        kGm.SetGlobalBuffer((__gm__ float*)key, totalBatchHeads * headStride);
        vGm.SetGlobalBuffer((__gm__ float*)value, totalBatchHeads * headStride);
        oGm.SetGlobalBuffer((__gm__ float*)output, totalBatchHeads * headStride);

        // workspace for attention scores: totalBatchHeads * seqLen * seqLen
        wGm.SetGlobalBuffer((__gm__ float*)workspace, totalBatchHeads * seqLen * seqLen);

        // Align headDim to 32 bytes (8 floats)
        uint32_t alignedHeadDim = ((headDim + 7) / 8) * 8;
        uint32_t alignedSeqLen = ((seqLen + 7) / 8) * 8;

        // Buffers for one query row, one key row, score row, value row, output row
        pipe.InitBuffer(qBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(scoreBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, alignedSeqLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t bh = startTask; bh < endTask; bh++) {
            ProcessOneHead(bh);
        }
    }

private:
    __aicore__ inline void ProcessOneHead(uint32_t bh)
    {
        uint32_t headStride = seqLen * headDim;
        uint32_t baseOffset = bh * headStride;
        uint32_t scoreBase = bh * seqLen * seqLen;
        float scale = 1.0f;
        // Compute sqrt(headDim) for scaling
        // scale = 1/sqrt(headDim)
        float invScale = 1.0f;
        {
            float hd = (float)headDim;
            // approximate 1/sqrt using iteration
            float s = 1.0f;
            for (int iter = 0; iter < 10; iter++) {
                s = s * 0.5f * (3.0f - hd * s * s);
            }
            invScale = s;
        }

        uint32_t alignedHeadDim = ((headDim + 7) / 8) * 8;
        uint32_t alignedSeqLen = ((seqLen + 7) / 8) * 8;

        for (uint32_t i = 0; i < seqLen; i++) {
            // Load query[i]
            AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
            AscendC::DataCopy(qLocal, qGm[baseOffset + i * headDim], alignedHeadDim);
            pipe.EnQue(qBuf, qLocal);
            qLocal = qBuf.DeQue<float>();

            AscendC::LocalTensor<float> scoreLocal = scoreBuf.AllocTensor<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();

            // Initialize scores to -inf (large negative)
            float negInf = -3.402823e+38f;
            AscendC::Duplicate(scoreLocal, negInf, alignedSeqLen);

            // Window boundaries
            uint32_t wStart = 0;
            if (i >= windowSize) wStart = i - windowSize;
            uint32_t wEnd = i + windowSize;
            if (wEnd > seqLen) wEnd = seqLen;

            // Compute dot products for keys in window
            for (uint32_t j = wStart; j < wEnd; j++) {
                AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
                AscendC::DataCopy(kLocal, kGm[baseOffset + j * headDim], alignedHeadDim);
                pipe.EnQue(kBuf, kLocal);
                kLocal = kBuf.DeQue<float>();

                // Dot product: q[i] . k[j]
                AscendC::Mul(tmpLocal, qLocal, kLocal, alignedHeadDim);
                // Sum reduction
                float dotVal = 0.0f;
                for (uint32_t d = 0; d < headDim; d++) {
                    dotVal += tmpLocal.GetValue(d);
                }
                dotVal *= invScale;
                scoreLocal.SetValue(j, dotVal);
                kBuf.FreeTensor(kLocal);
            }

            // Softmax over the valid window
            // Find max
            float maxVal = -3.402823e+38f;
            for (uint32_t j = wStart; j < wEnd; j++) {
                float v = scoreLocal.GetValue(j);
                if (v > maxVal) maxVal = v;
            }
            // Exp and sum
            float expSum = 0.0f;
            for (uint32_t j = wStart; j < wEnd; j++) {
                float v = scoreLocal.GetValue(j);
                // Compute exp manually using AscendC or scalar
                float e = 0.0f;
                float arg = v - maxVal;
                // Use simple exp approximation or built-in
                // We'll use a Taylor-ish approach or set values
                // Actually let's store shifted values and use Exp intrinsic
                scoreLocal.SetValue(j, arg);
            }
            // Set non-window to very negative so exp ~ 0
            // Use AscendC::Exp on the score buffer
            AscendC::Exp(tmpLocal, scoreLocal, alignedSeqLen);

            expSum = 0.0f;
            for (uint32_t j = wStart; j < wEnd; j++) {
                expSum += tmpLocal.GetValue(j);
            }
            float invSum = 1.0f / expSum;

            // Weighted sum of values
            AscendC::LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
            AscendC::Duplicate(outLocal, 0.0f, alignedHeadDim);

            for (uint32_t j = wStart; j < wEnd; j++) {
                float weight = tmpLocal.GetValue(j) * invSum;
                AscendC::LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
                AscendC::DataCopy(vLocal, vGm[baseOffset + j * headDim], alignedHeadDim);
                pipe.EnQue(vBuf, vLocal);
                vLocal = vBuf.DeQue<float>();

                // outLocal += weight * vLocal
                AscendC::Muls(vLocal, vLocal, weight, alignedHeadDim);
                AscendC::Add(outLocal, outLocal, vLocal, alignedHeadDim);

                vBuf.FreeTensor(vLocal);
            }

            AscendC::DataCopy(oGm[baseOffset + i * headDim], outLocal, alignedHeadDim);

            outBuf.FreeTensor(outLocal);
            tmpBuf.FreeTensor(tmpLocal);
            scoreBuf.FreeTensor(scoreLocal);
            qBuf.FreeTensor(qLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qBuf, kBuf, scoreBuf, vBuf, outBuf, tmpBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, oGm, wGm;
    uint32_t seqLen, headDim, windowSize, totalBatchHeads;
    uint32_t tasksPerBlock, startTask, endTask;
};

extern "C" __global__ __aicore__ void sparse_attention_custom(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSparseAttention op;
    op.Init(query, key, value, output, workspace,
            tiling_data.batchSize, tiling_data.numHeads, tiling_data.seqLen,
            tiling_data.headDim, tiling_data.windowSize, tiling_data.totalBatchHeads);
    op.Process();
}
