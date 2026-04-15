
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAvgPoolGeluScaleMax {
public:
    __aicore__ inline KernelAvgPoolGeluScaleMax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t inputLength,
                                 uint32_t poolKernelSize, float scaleFactor)
    {
        this->batchSize = batchSize;
        this->inputLength = inputLength;
        this->poolKernelSize = poolKernelSize;
        this->scaleFactor = scaleFactor;
        this->pooledLength = inputLength / poolKernelSize;

        // Each block processes a subset of batch items
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->batchPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->batchStart = blockIdx * this->batchPerBlock;
        this->batchEnd = batchStart + batchPerBlock;
        if (this->batchEnd > batchSize) {
            this->batchEnd = batchSize;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * inputLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize);

        // Allocate buffers for processing one row at a time
        // We need buffer for input row, pooled result, gelu result
        uint32_t alignedInput = (inputLength + 7) / 8 * 8;
        uint32_t alignedPooled = (pooledLength + 7) / 8 * 8;

        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedInput * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedPooled * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t b = batchStart; b < batchEnd; b++) {
            CopyIn(b);
            Compute(b);
            CopyOut(b);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[batchIdx * inputLength], (inputLength + 7) / 8 * 8);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

        uint32_t alignedPooled = (pooledLength + 7) / 8 * 8;

        // Average pooling: for each pooled element, sum poolKernelSize elements and divide
        // We'll do this by extracting segments and reducing
        float invPool = 1.0f / (float)poolKernelSize;

        // Initialize outLocal to zero
        AscendC::Duplicate<float>(outLocal, 0.0f, alignedPooled);

        // For avg pool: accumulate poolKernelSize elements
        for (uint32_t k = 0; k < poolKernelSize; k++) {
            // We need to gather elements at positions k, k+poolKernelSize, k+2*poolKernelSize, ...
            // Actually avg pool with kernel_size over contiguous: pool[i] = mean(x[i*K .. (i+1)*K-1])
            // So pool[i] = sum(x[i*K+0], x[i*K+1], ..., x[i*K+K-1]) / K
            // We can add x[offset] where offset shifts by 1 each iteration
            // But we need strided access or copy

            // For each pool output i, we add x[i*poolKernelSize + k]
            // This is a gather operation - we'll manually handle it
            // Since AscendC may not directly support strided gather, we iterate
        }

        // Simpler approach: iterate over pooled outputs in chunks
        // Use a different strategy: for each k in [0, poolKernelSize), add the k-th element of each pool window
        // x[i*poolKernelSize + k] for all i
        // This can be done if we treat it as strided access

        // Reset to zero
        AscendC::Duplicate<float>(outLocal, 0.0f, alignedPooled);

        // Accumulate
        for (uint32_t k = 0; k < poolKernelSize; k++) {
            for (uint32_t i = 0; i < pooledLength; i++) {
                // This is scalar but correct
                float val = xLocal.GetValue(i * poolKernelSize + k);
                float cur = outLocal.GetValue(i);
                outLocal.SetValue(i, cur + val);
            }
        }

        // Divide by poolKernelSize
        AscendC::Muls(outLocal, outLocal, invPool, alignedPooled);

        // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Use AscendC built-in if available, otherwise approximate
        // We will use the tanh approximation
        for (uint32_t i = 0; i < pooledLength; i++) {
            float val = outLocal.GetValue(i);
            float x3 = val * val * val;
            float inner = 0.7978845608f * (val + 0.044715f * x3);
            // tanh approximation or compute
            float ep = 1.0f;
            float en = 1.0f;
            float t = inner;
            // Simple tanh: (exp(2x)-1)/(exp(2x)+1)
            // Use limited precision
            float e2x = 1.0f;
            float term = 2.0f * t;
            float factorial = 1.0f;
            // exp(2t) via series - but this is slow. Let's use a simpler approach
            // Clamp and use rational approximation
            if (t > 4.0f) {
                ep = 1.0f;
            } else if (t < -4.0f) {
                ep = -1.0f;
            } else {
                // exp(2t)
                float x2t = 2.0f * t;
                float exp_val = 1.0f;
                float power = 1.0f;
                for (int j = 1; j <= 12; j++) {
                    power *= x2t / (float)j;
                    exp_val += power;
                }
                ep = (exp_val - 1.0f) / (exp_val + 1.0f);
            }
            float gelu_val = 0.5f * val * (1.0f + ep);
            outLocal.SetValue(i, gelu_val);
        }

        // Scale
        AscendC::Muls(outLocal, outLocal, scaleFactor, alignedPooled);

        // Max reduction
        float maxVal = outLocal.GetValue(0);
        for (uint32_t i = 1; i < pooledLength; i++) {
            float v = outLocal.GetValue(i);
            if (v > maxVal) maxVal = v;
        }

        // Store the scalar max result
        outLocal.SetValue(0, maxVal);

        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        // Copy just 1 element, but must be aligned to 32 bytes (8 floats)
        AscendC::DataCopy(yGm[batchIdx], outLocal, 8);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t inputLength;
    uint32_t poolKernelSize;
    uint32_t pooledLength;
    float scaleFactor;
    uint32_t batchPerBlock;
    uint32_t batchStart;
    uint32_t batchEnd;
};

extern "C" __global__ __aicore__ void matmul_avg_pool_gelu_scale_max_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAvgPoolGeluScaleMax op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.inputLength,
            tiling_data.poolKernelSize, tiling_data.scaleFactor);
    op.Process();
}
