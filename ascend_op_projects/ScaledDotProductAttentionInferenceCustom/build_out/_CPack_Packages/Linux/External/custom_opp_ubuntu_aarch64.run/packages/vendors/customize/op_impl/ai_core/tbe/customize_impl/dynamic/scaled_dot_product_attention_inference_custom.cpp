
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelScaledDotProductAttention {
public:
    __aicore__ inline KernelScaledDotProductAttention() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t seqLen, uint32_t dModel)
    {
        this->batchSize = batchSize;
        this->seqLen = seqLen;
        this->dModel = dModel;
        this->scale = 1.0f / AscendC::Sqrt(static_cast<float>(dModel));

        uint32_t totalElements = batchSize * seqLen * dModel;

        qGm.SetGlobalBuffer((__gm__ float *)q, totalElements);
        kGm.SetGlobalBuffer((__gm__ float *)k, totalElements);
        vGm.SetGlobalBuffer((__gm__ float *)v, totalElements);
        oGm.SetGlobalBuffer((__gm__ float *)o, totalElements);
        workGm.SetGlobalBuffer((__gm__ float *)workspace, batchSize * seqLen * seqLen);

        // Align dModel and seqLen to 8 for float (32 bytes / 4 bytes per float = 8)
        uint32_t dModelAligned = ((dModel + 7) / 8) * 8;
        uint32_t seqLenAligned = ((seqLen + 7) / 8) * 8;

        // Allocate buffers
        // We need: qRow (dModelAligned), kRow (dModelAligned), scores (seqLenAligned),
        // temp (dModelAligned or seqLenAligned), vRow (dModelAligned), outRow (dModelAligned)
        pipe.InitBuffer(qBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(kBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(scoresBuf, BUFFER_NUM, seqLenAligned * sizeof(float));
        pipe.InitBuffer(tempBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(vBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(outBuf, BUFFER_NUM, dModelAligned * sizeof(float));
        pipe.InitBuffer(maxBuf, BUFFER_NUM, seqLenAligned * sizeof(float));
        pipe.InitBuffer(sumBuf, BUFFER_NUM, seqLenAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t b = 0; b < batchSize; b++) {
            for (uint32_t i = 0; i < seqLen; i++) {
                ComputeRow(b, i);
            }
        }
    }

private:
    __aicore__ inline void ComputeRow(uint32_t b, uint32_t queryIdx)
    {
        uint32_t dModelAligned = ((dModel + 7) / 8) * 8;
        uint32_t seqLenAligned = ((seqLen + 7) / 8) * 8;

        uint32_t batchOffsetQ = b * seqLen * dModel;

        // Load query row into local memory
        AscendC::LocalTensor<float> qLocal = qBuf.AllocTensor<float>();
        AscendC::DataCopy(qLocal, qGm[batchOffsetQ + queryIdx * dModel], dModelAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Compute dot products: scores[j] = Q[i] . K[j] for all j
        AscendC::LocalTensor<float> scoresLocal = scoresBuf.AllocTensor<float>();

        // Initialize scores to zero
        AscendC::Duplicate(scoresLocal, 0.0f, seqLenAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> kLocal = kBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> tempLocal = tempBuf.AllocTensor<float>();

        for (uint32_t j = 0; j < seqLen; j++) {
            // Load K[j] row
            AscendC::DataCopy(kLocal, kGm[batchOffsetQ + j * dModel], dModelAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Element-wise multiply Q[i] * K[j]
            AscendC::Mul(tempLocal, qLocal, kLocal, dModelAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Sum reduction to get dot product
            float dotProduct = 0.0f;
            // Use ReduceSum if available, otherwise manual
            // We'll do a simple loop for correctness
            uint32_t reduceLen = dModelAligned;
            while (reduceLen > 1) {
                uint32_t half = reduceLen / 2;
                if (half >= 8) {
                    AscendC::Add(tempLocal, tempLocal, tempLocal[half], half);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                reduceLen = half;
                if (reduceLen < 8) break;
            }
            // Copy partial result to GM workspace and read back
            // Actually let's just store score to workspace GM
            workGm.SetValue(b * seqLen * seqLen + queryIdx * seqLen + j, 0.0f);
            AscendC::PipeBarrier<PIPE_ALL>();

            // For simplicity, copy the first 8 elements to compute sum
            float partial = 0.0f;
            // We'll use a different approach - copy tempLocal to GM and sum there
            // Actually, let's use SetValue approach through workspace
            
            // Store the reduced partial sums (first 8 elements after reduction)
            AscendC::DataCopy(workGm[b * seqLen * seqLen + queryIdx * seqLen + j], tempLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        kBuf.FreeTensor(kLocal);
        tempBuf.FreeTensor(tempLocal);

        // Now we need to properly compute dot products
        // Let's use a simpler approach: compute in tiles through workspace
        // Read back scores from workspace, apply scale, softmax, then weighted sum of V

        // Actually, let me redo this with a cleaner approach
        // We'll compute scores by reading them back from workspace reduction
        
        // For each j, sum the 8 partial values in workspace
        AscendC::LocalTensor<float> maxLocal = maxBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.AllocTensor<float>();

        // Read all partial scores and finalize
        AscendC::LocalTensor<float> tempLocal2 = tempBuf.AllocTensor<float>();
        
        for (uint32_t j = 0; j < seqLen; j++) {
            AscendC::DataCopy(tempLocal2, workGm[b * seqLen * seqLen + queryIdx * seqLen + j], 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            // Sum first 8 elements manually through reduction
            float s = 0.0f;
            // We can't easily get scalar from local tensor, so store partial sum
            AscendC::Add(tempLocal2, tempLocal2, tempLocal2[4], 4);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(tempLocal2, tempLocal2, tempLocal2[2], 2);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(tempLocal2, tempLocal2, tempLocal2[1], 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            // tempLocal2[0] now has the sum, multiply by scale
            AscendC::Muls(tempLocal2, tempLocal2, this->scale, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
            // Store score[j]
            // Copy single value to scoresLocal position
            // Since we can't index scoresLocal by j easily for single element,
            // we store to workspace and reload
            AscendC::DataCopy(workGm[b * seqLen * seqLen + queryIdx * seqLen + j], tempLocal2, 8);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        tempBuf.FreeTensor(tempLocal2);

        // Load all scores
        AscendC::DataCopy(scoresLocal, workGm[b * seqLen * seqLen + queryIdx * seqLen], seqLenAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Softmax: find max, subtract, exp, normalize
        // Find max using ReduceMax
        AscendC::ReduceMax(maxLocal, scoresLocal, seqLenAligned, false);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Subtract max from all scores
        AscendC::Adds(scoresLocal, scoresLocal, -1.0f * maxLocal.GetValue(0), seqLenAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Exp
        AscendC::Exp(scoresLocal, scoresLocal, seqLenAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Zero out elements beyond seqLen if seqLenAligned > seqLen
        for (uint32_t j = seqLen; j < seqLenAligned; j++) {
            scoresLocal.SetValue(j, 0.0f);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // Sum
        AscendC::ReduceSum(sumLocal, scoresLocal, seqLenAligned, false);
        AscendC::PipeBarrier<PIPE_ALL>();

        float sumVal = sumLocal.GetValue(0);
        if (sumVal > 0.0f) {
            AscendC::Muls(scoresLocal, scoresLocal, 1.0f / sumVal, seqLenAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        maxBuf.FreeTensor(maxLocal);
        sumBuf.FreeTensor(sumLocal);

        // Store softmax scores to workspace
        AscendC::DataCopy(workGm[b * seqLen * seqLen + queryIdx * seqLen], scoresLocal, seqLenAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        scoresBuf.FreeTensor(scoresLocal);
        qBuf.FreeTensor(qLocal);

        // Compute output[i] = sum_j scores[j] * V[j]
        AscendC::LocalTensor<float> outLocal = outBuf.AllocTensor<float>();
        AscendC::Duplicate(outLocal, 0.0f, dModelAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::LocalTensor<float> vLocal = vBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> tempLocal3 = tempBuf.AllocTensor<float>();

        for (uint32_t j = 0; j < seqLen; j++) {
            float score_j = workGm.GetValue(b * seqLen * seqLen + queryIdx * seqLen + j);
            if (score_j == 0.0f) continue;

            AscendC::DataCopy(vLocal, vGm[batchOffsetQ + j * dModel], dModelAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Muls(tempLocal3, vLocal, score_j, dModelAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Add(outLocal, outLocal, tempLocal3, dModelAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // Store output
        AscendC::DataCopy(oGm[batchOffsetQ + queryIdx * dModel], outLocal, dModelAligned);
        AscendC::PipeBarrier<PIPE_ALL>();

        vBuf.FreeTensor(vLocal);
        tempBuf.FreeTensor(tempLocal3);
        outBuf.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> qBuf, kBuf, scoresBuf, tempBuf, vBuf, maxBuf, sumBuf;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outBuf;
    AscendC::GlobalTensor<float> qGm, kGm, vGm, oGm, workGm;
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t dModel;
    float scale;
};

extern "C" __global__ __aicore__ void scaled_dot_product_attention_inference_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaledDotProductAttention op;
    op.Init(q, k, v, o, workspace,
            tiling_data.batchSize, tiling_data.seqLen, tiling_data.dModel);
    op.Process();
}
