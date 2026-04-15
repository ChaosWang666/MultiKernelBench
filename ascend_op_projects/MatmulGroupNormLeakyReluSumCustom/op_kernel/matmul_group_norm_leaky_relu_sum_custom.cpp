
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmulGroupNormLeakyReluSum {
public:
    __aicore__ inline KernelMatmulGroupNormLeakyReluSum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR gamma, GM_ADDR beta, GM_ADDR z,
                                 uint32_t batchSize, uint32_t inputSize, uint32_t hiddenSize, uint32_t numGroups,
                                 float eps, float negativeSlope)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->numGroups = numGroups;
        this->eps = eps;
        this->negativeSlope = negativeSlope;
        this->channelsPerGroup = hiddenSize / numGroups;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Each block processes one sample in the batch
        this->sampleIdx = blockIdx;

        xGm.SetGlobalBuffer((__gm__ float *)x + sampleIdx * inputSize, inputSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, hiddenSize * inputSize);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, hiddenSize);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, hiddenSize);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, hiddenSize);
        zGm.SetGlobalBuffer((__gm__ float *)z + sampleIdx * hiddenSize, hiddenSize);

        // We process the hidden dimension in tiles
        // Each tile handles channelsPerGroup elements for one group at a time
        uint32_t tileSize = this->channelsPerGroup;
        // Align tile size to 32 bytes (8 floats)
        uint32_t alignedTileSize = (tileSize + 7) / 8 * 8;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, inputSize * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, alignedTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf1, BUFFER_NUM, alignedTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf2, BUFFER_NUM, alignedTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf3, BUFFER_NUM, alignedTileSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Load x for this sample once
        CopyInX();

        // Process each group
        for (uint32_t g = 0; g < numGroups; g++) {
            ComputeGroup(g);
            CopyOut(g);
        }
    }

private:
    __aicore__ inline void CopyInX()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm, inputSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeGroup(uint32_t groupIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        uint32_t startCh = groupIdx * channelsPerGroup;
        uint32_t alignedCPG = (channelsPerGroup + 7) / 8 * 8;

        // Allocate output and temp buffers
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp3 = tmpBuf3.AllocTensor<float>();

        // Step 1: Matmul - compute linear output for channels in this group
        // For each channel c in [startCh, startCh + channelsPerGroup):
        //   linear[c] = dot(x, weight[c]) + bias[c]
        // We compute this channel by channel and store in zLocal

        for (uint32_t c = 0; c < channelsPerGroup; c++) {
            uint32_t ch = startCh + c;
            float dotVal = 0.0f;
            // Compute dot product in tiles
            // We'll do vector multiply and reduce
            // For simplicity with large inputSize, tile the dot product
            uint32_t tileLen = 1024;
            float accumulator = 0.0f;
            for (uint32_t offset = 0; offset < inputSize; offset += tileLen) {
                uint32_t curLen = tileLen;
                if (offset + curLen > inputSize) {
                    curLen = inputSize - offset;
                }
                uint32_t alignedLen = (curLen + 7) / 8 * 8;
                // We need to load weight segment - but we can't easily load from weightGm in tiles here
                // Instead, let's do scalar computation
            }
            // Fallback: scalar dot product
            // This is slow but correct. For production, we'd use a matmul intrinsic.
        }

        // Actually, let's simplify: compute the full hidden vector using scalar ops for correctness
        // Then apply group norm, leaky relu, and sum (x2)
        
        // Compute linear output for this group's channels
        for (uint32_t c = 0; c < channelsPerGroup; c++) {
            uint32_t ch = startCh + c;
            float val = 0.0f;
            for (uint32_t k = 0; k < inputSize; k++) {
                val += xLocal.GetValue(k) * weightGm.GetValue(ch * inputSize + k);
            }
            val += biasGm.GetValue(ch);
            zLocal.SetValue(c, val);
        }

        // Step 2: Group Normalization over this group's channels
        // Compute mean
        float mean = 0.0f;
        for (uint32_t c = 0; c < channelsPerGroup; c++) {
            mean += zLocal.GetValue(c);
        }
        mean /= (float)channelsPerGroup;

        // Compute variance
        float var = 0.0f;
        for (uint32_t c = 0; c < channelsPerGroup; c++) {
            float diff = zLocal.GetValue(c) - mean;
            var += diff * diff;
        }
        var /= (float)channelsPerGroup;

        float invStd = 1.0f / sqrtf(var + eps);

        // Normalize, scale, shift, leaky relu, and double (sum with self)
        for (uint32_t c = 0; c < channelsPerGroup; c++) {
            uint32_t ch = startCh + c;
            float normalized = (zLocal.GetValue(c) - mean) * invStd;
            float scaled = normalized * gammaGm.GetValue(ch) + betaGm.GetValue(ch);
            // Leaky ReLU
            float activated = scaled >= 0.0f ? scaled : scaled * negativeSlope;
            // Sum with self (x + x = 2*x)
            float result = activated * 2.0f;
            zLocal.SetValue(c, result);
        }

        // Pad remaining elements if alignedCPG > channelsPerGroup
        for (uint32_t c = channelsPerGroup; c < alignedCPG; c++) {
            zLocal.SetValue(c, 0.0f);
        }

        outQueueZ.EnQue(zLocal);
        tmpBuf1.FreeTensor(tmp1);
        tmpBuf2.FreeTensor(tmp2);
        tmpBuf3.FreeTensor(tmp3);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t groupIdx)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        uint32_t startCh = groupIdx * channelsPerGroup;
        AscendC::DataCopy(zGm[startCh], zLocal, (channelsPerGroup + 7) / 8 * 8);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf1, tmpBuf2, tmpBuf3;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> gammaGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint32_t numGroups;
    uint32_t channelsPerGroup;
    uint32_t sampleIdx;
    float eps;
    float negativeSlope;
};

extern "C" __global__ __aicore__ void matmul_group_norm_leaky_relu_sum_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR gamma, GM_ADDR beta, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulGroupNormLeakyReluSum op;
    op.Init(x, weight, bias, gamma, beta, z,
            tiling_data.batchSize, tiling_data.inputSize, tiling_data.hiddenSize,
            tiling_data.numGroups, tiling_data.eps, tiling_data.negativeSlope);
    op.Process();
}
