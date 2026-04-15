
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCrossAttention {
public:
    __aicore__ inline KernelCrossAttention() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output,
                                 GM_ADDR workspace,
                                 uint32_t seqLenQ, uint32_t seqLenKV, uint32_t headDim,
                                 uint32_t totalBatchHeads)
    {
        this->seqLenQ = seqLenQ;
        this->seqLenKV = seqLenKV;
        this->headDim = headDim;
        this->totalBatchHeads = totalBatchHeads;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        this->startHead = blockIdx;
        this->headStride = blockNum;

        uint32_t qStride = seqLenQ * headDim;
        uint32_t kvStride = seqLenKV * headDim;

        qGm.SetGlobalBuffer((__gm__ float*)query, totalBatchHeads * qStride);
        kGm.SetGlobalBuffer((__gm__ float*)key, totalBatchHeads * kvStride);
        vGm.SetGlobalBuffer((__gm__ float*)value, totalBatchHeads * kvStride);
        outGm.SetGlobalBuffer((__gm__ float*)output, totalBatchHeads * qStride);
        workGm.SetGlobalBuffer((__gm__ float*)workspace, totalBatchHeads * seqLenQ * seqLenKV);

        // Allocate buffers for one row of Q, one row of scores, K tile, V tile, output row
        uint32_t alignedHeadDim = (headDim + 7) / 8 * 8;
        uint32_t alignedSeqLenKV = (seqLenKV + 7) / 8 * 8;

        pipe.InitBuffer(qBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, alignedSeqLenKV * sizeof(float));
        pipe.InitBuffer(scoreBuf, BUFFER_NUM, alignedSeqLenKV * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, alignedHeadDim * sizeof(float));
        pipe.InitBuffer(maxBuf, BUFFER_NUM, alignedSeqLenKV * sizeof(float));
        pipe.InitBuffer(sumBuf, BUFFER_NUM, alignedSeqLenKV * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        float scale = 1.0f / AscendC::Sqrt((float)headDim);
        uint32_t alignedHeadDim = (headDim + 7) / 8 * 8;
        uint32_t alignedSeqLenKV = (seqLenKV + 7) / 8 * 8;

        for (uint32_t h = startHead; h < totalBatchHeads; h += headStride) {
            uint32_t qOffset = h * seqLenQ * headDim;
            uint32_t kvOffset = h * seqLenKV * headDim;

            for (uint32_t qi = 0; qi < seqLenQ; qi++) {
                // Load Q row
                AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
                AscendC::DataCopy(qLocal, qGm[qOffset + qi * headDim], alignedHeadDim);
                pipe_barrier(PIPE_ALL);

                // Compute scores: Q[qi] dot K[ki] for all ki
                AscendC::LocalTensor<float> scoreLocal = scoreBuf.AllocTensor<float>();

                for (uint32_t ki = 0; ki < seqLenKV; ki++) {
                    // Load K row
                    AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
                    AscendC::DataCopy(kLocal, kGm[kvOffset + ki * headDim], alignedHeadDim);
                    pipe_barrier(PIPE_ALL);

                    // Dot product: element-wise multiply then reduce
                    AscendC::LocalTensor<float> tempLocal = maxBuf.AllocTensor<float>();
                    AscendC::Mul(tempLocal, qLocal, kLocal, alignedHeadDim);
                    pipe_barrier(PIPE_ALL);

                    // Sum reduction
                    float dotVal = 0.0f;
                    for (uint32_t d = 0; d < headDim; d++) {
                        dotVal += tempLocal.GetValue(d);
                    }
                    scoreLocal.SetValue(ki, dotVal * scale);

                    maxBuf.FreeTensor(tempLocal);
                    kBuf.FreeTensor(kLocal);
                }
                pipe_barrier(PIPE_ALL);

                // Softmax over scores
                // Find max
                float maxVal = scoreLocal.GetValue(0);
                for (uint32_t ki = 1; ki < seqLenKV; ki++) {
                    float v = scoreLocal.GetValue(ki);
                    if (v > maxVal) maxVal = v;
                }

                // Exp and sum
                float sumExp = 0.0f;
                AscendC::LocalTensor<float> sumLocal = sumBuf.AllocTensor<float>();
                for (uint32_t ki = 0; ki < seqLenKV; ki++) {
                    float v = AscendC::Exp(scoreLocal.GetValue(ki) - maxVal);
                    sumLocal.SetValue(ki, v);
                    sumExp += v;
                }
                pipe_barrier(PIPE_ALL);

                float invSum = 1.0f / sumExp;
                for (uint32_t ki = 0; ki < seqLenKV; ki++) {
                    sumLocal.SetValue(ki, sumLocal.GetValue(ki) * invSum);
                }
                pipe_barrier(PIPE_ALL);

                // Weighted sum of V
                AscendC::LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
                for (uint32_t d = 0; d < alignedHeadDim; d++) {
                    outLocal.SetValue(d, 0.0f);
                }
                pipe_barrier(PIPE_ALL);

                for (uint32_t ki = 0; ki < seqLenKV; ki++) {
                    float w = sumLocal.GetValue(ki);
                    AscendC::LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
                    AscendC::DataCopy(vLocal, vGm[kvOffset + ki * headDim], alignedHeadDim);
                    pipe_barrier(PIPE_ALL);

                    for (uint32_t d = 0; d < headDim; d++) {
                        outLocal.SetValue(d, outLocal.GetValue(d) + w * vLocal.GetValue(d));
                    }
                    pipe_barrier(PIPE_ALL);
                    vBuf.FreeTensor(vLocal);
                }
                pipe_barrier(PIPE_ALL);

                // Write output
                AscendC::DataCopy(outGm[qOffset + qi * headDim], outLocal, alignedHeadDim);
                pipe_barrier(PIPE_ALL);

                outBuf.FreeTensor(outLocal);
                sumBuf.FreeTensor(sumLocal);
                scoreBuf.FreeTensor(scoreLocal);
                qBuf.FreeTensor(qLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qBuf, kBuf, vBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outBuf;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> scoreBuf, maxBuf, sumBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm, workGm;
    uint32_t seqLenQ, seqLenKV, headDim, totalBatchHeads;
    uint32_t startHead, headStride;
};

extern "C" __global__ __aicore__ void cross_attention_custom(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossAttention op;
    op.Init(query, key, value, output, workspace,
            tiling_data.seqLenQ, tiling_data.seqLenKV, tiling_data.headDim,
            tiling_data.totalBatchHeads);
    op.Process();
}
