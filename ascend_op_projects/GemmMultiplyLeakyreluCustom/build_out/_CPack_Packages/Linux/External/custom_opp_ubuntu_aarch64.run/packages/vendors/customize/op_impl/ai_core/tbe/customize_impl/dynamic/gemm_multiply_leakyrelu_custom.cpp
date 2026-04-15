
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmMultiplyLeakyrelu {
public:
    __aicore__ inline KernelGemmMultiplyLeakyrelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, float multiplier, float negativeSlope)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->multiplier = multiplier;
        this->negativeSlope = negativeSlope;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        // Multiply by multiplier
        AscendC::Muls(zLocal, xLocal, this->multiplier, this->tileLength);

        // LeakyReLU: if x > 0 then x, else negativeSlope * x
        // Compute negativeSlope * zLocal into tmpLocal
        AscendC::Muls(tmpLocal, zLocal, this->negativeSlope, this->tileLength);

        // Use Max to get max(zLocal, 0) - but we need LeakyReLU
        // LeakyReLU(x) = max(x, 0) + negativeSlope * min(x, 0)
        // = max(x, 0) + min(negativeSlope * x, 0)  (when negativeSlope > 0)
        // We can use: result = select(x >= 0, x, negativeSlope * x)
        // AscendC approach: use LeakyRelu intrinsic if available, or manual

        // Manual approach using Maxs and Mins:
        // pos_part = max(zLocal, 0)
        // neg_part = min(zLocal, 0) * negativeSlope
        // result = pos_part + neg_part

        AscendC::LocalTensor<float> posLocal = xLocal; // reuse xLocal buffer

        // pos_part = max(zLocal, scalar 0)
        AscendC::Maxs(posLocal, zLocal, (float)0.0f, this->tileLength);

        // neg_part = min(zLocal, scalar 0) stored in tmpLocal
        AscendC::Mins(tmpLocal, zLocal, (float)0.0f, this->tileLength);
        AscendC::Muls(tmpLocal, tmpLocal, this->negativeSlope, this->tileLength);

        // result = pos_part + neg_part
        AscendC::Add(zLocal, posLocal, tmpLocal, this->tileLength);

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float multiplier;
    float negativeSlope;
};

extern "C" __global__ __aicore__ void gemm_multiply_leakyrelu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmMultiplyLeakyrelu op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.multiplier, tiling_data.negativeSlope);
    op.Process();
}
