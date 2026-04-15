
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv3dLeakyReluSumClampGelu {
public:
    __aicore__ inline KernelConv3dLeakyReluSumClampGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sumTensor, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t outChannels, uint32_t spatialSize)
    {
        this->outChannels = outChannels;
        this->spatialSize = spatialSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        sumGm.SetGlobalBuffer((__gm__ float *)sumTensor, outChannels);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // LeakyReLU: y = x > 0 ? x : 0.2 * x
        // Compute 0.2 * x
        AscendC::Muls(tmp1, xLocal, 0.2f, this->tileLength);
        // Max(0.2*x, x) gives LeakyReLU for negative_slope=0.2
        AscendC::Max(zLocal, xLocal, tmp1, this->tileLength);

        // Add sum_tensor with broadcasting: sum_tensor shape is [C,1,1,1]
        // x shape after conv is [N, C, D, H, W], flattened
        // Global offset for this block
        uint32_t globalOffset = this->blockLength * AscendC::GetBlockIdx() + progress * this->tileLength;
        for (uint32_t i = 0; i < this->tileLength; i++) {
            uint32_t globalIdx = globalOffset + i;
            // channel index: (globalIdx / spatialSize) % outChannels
            uint32_t channelIdx = (globalIdx / this->spatialSize) % this->outChannels;
            tmp1.SetValue(i, zLocal.GetValue(i) + sumGm.GetValue(channelIdx));
        }

        // Clamp to [-1, 1]
        float minVal = -1.0f;
        float maxVal = 1.0f;
        // clamp min: max(x, -1)
        AscendC::Maxs(tmp2, tmp1, minVal, this->tileLength);
        // clamp max: min(x, 1)
        AscendC::Mins(tmp1, tmp2, maxVal, this->tileLength);

        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // But we can use: x * 0.5 * (1 + erf(x / sqrt(2)))
        // Use simpler approach with available ops

        // Compute x^2
        AscendC::Mul(tmp2, tmp1, tmp1, this->tileLength);
        // Compute x^3
        AscendC::Mul(zLocal, tmp2, tmp1, this->tileLength);
        // 0.044715 * x^3
        AscendC::Muls(zLocal, zLocal, 0.044715f, this->tileLength);
        // x + 0.044715*x^3
        AscendC::Add(zLocal, tmp1, zLocal, this->tileLength);
        // sqrt(2/pi) = 0.7978845608
        AscendC::Muls(zLocal, zLocal, 0.7978845608f, this->tileLength);
        // tanh
        for (uint32_t i = 0; i < this->tileLength; i++) {
            float val = zLocal.GetValue(i);
            // tanh approximation
            float ep = 1.0f;
            float en = 1.0f;
            float absval = val > 0 ? val : -val;
            // Use polynomial approximation for exp for speed, or direct computation
            // Simple: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
            float e2x;
            float tv = 2.0f * val;
            // Clamp to avoid overflow
            if (tv > 10.0f) tv = 10.0f;
            if (tv < -10.0f) tv = -10.0f;
            // exp approximation using iterations
            e2x = 1.0f + tv + tv*tv*0.5f + tv*tv*tv/6.0f + tv*tv*tv*tv/24.0f + tv*tv*tv*tv*tv/120.0f + tv*tv*tv*tv*tv*tv/720.0f;
            if (e2x < 0.0f) e2x = 0.0001f;
            float tanhVal = (e2x - 1.0f) / (e2x + 1.0f);
            tmp2.SetValue(i, tanhVal);
        }
        // 1 + tanh(...)
        AscendC::Adds(zLocal, tmp2, 1.0f, this->tileLength);
        // 0.5 * x
        AscendC::Muls(tmp2, tmp1, 0.5f, this->tileLength);
        // result = 0.5*x * (1 + tanh(...))
        AscendC::Mul(zLocal, tmp2, zLocal, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> sumGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t outChannels;
    uint32_t spatialSize;
};

extern "C" __global__ __aicore__ void conv3d_leaky_relu_sum_clamp_gelu_custom(GM_ADDR x, GM_ADDR sumTensor, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dLeakyReluSumClampGelu op;
    op.Init(x, sumTensor, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.outChannels, tiling_data.spatialSize);
    op.Process();
}
