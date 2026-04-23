
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmScalingHardTanhGelu {
public:
    __aicore__ inline KernelGemmScalingHardTanhGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum,
                                  float scalingFactor, float hardtanhMin, float hardtanhMax)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->scalingFactor = scalingFactor;
        this->hardtanhMin = hardtanhMin;
        this->hardtanhMax = hardtanhMax;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        // Step 1: Scaling - x = x * scalingFactor
        AscendC::Muls(xLocal, xLocal, this->scalingFactor, this->tileLength);

        // Step 2: Hardtanh - x = clamp(x, hardtanhMin, hardtanhMax)
        AscendC::Duplicate(tmpLocal, this->hardtanhMin, this->tileLength);
        AscendC::Max(xLocal, xLocal, tmpLocal, this->tileLength);
        AscendC::Duplicate(tmpLocal, this->hardtanhMax, this->tileLength);
        AscendC::Min(xLocal, xLocal, tmpLocal, this->tileLength);

        // Step 3: GELU - 0.5 * x * (1 + erf(x / sqrt(2)))
        const float INV_SQRT_2 = 0.7071067811865475f;
        AscendC::Muls(tmpLocal, xLocal, INV_SQRT_2, this->tileLength);
        AscendC::Erf(tmpLocal, tmpLocal, this->tileLength);
        AscendC::Adds(tmpLocal, tmpLocal, 1.0f, this->tileLength);
        AscendC::Mul(tmpLocal, xLocal, tmpLocal, this->tileLength);
        AscendC::Muls(zLocal, tmpLocal, 0.5f, this->tileLength);

        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float scalingFactor;
    float hardtanhMin;
    float hardtanhMax;
};

extern "C" __global__ __aicore__ void gemm_scaling_hard_tanh_gelu_custom(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmScalingHardTanhGelu op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum,
            tiling_data.scalingFactor, tiling_data.hardtanhMin, tiling_data.hardtanhMax);
    op.Process();
}
