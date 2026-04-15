
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMishMish {
public:
    __aicore__ inline KernelMishMish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
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
    __aicore__ inline void ApplyMish(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src,
                                      AscendC::LocalTensor<float>& tmp1, AscendC::LocalTensor<float>& tmp2)
    {
        // mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // Step 1: tmp1 = exp(x)
        AscendC::Exp(tmp1, src, this->tileLength);
        // Step 2: tmp1 = 1 + exp(x)
        AscendC::Adds(tmp1, tmp1, (float)1.0, this->tileLength);
        // Step 3: tmp1 = ln(1 + exp(x)) = softplus(x)
        AscendC::Ln(tmp1, tmp1, this->tileLength);
        // Step 4: tmp2 = tanh(softplus(x))
        AscendC::Tanh(tmp2, tmp1, this->tileLength);
        // Step 5: dst = x * tanh(softplus(x))
        AscendC::Mul(dst, src, tmp2, this->tileLength);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // First Mish: zLocal = mish(xLocal)
        ApplyMish(zLocal, xLocal, tmp1, tmp2);
        // Second Mish: use xLocal as temp storage for result
        // Copy zLocal to xLocal first
        AscendC::DataCopy(xLocal, zLocal, this->tileLength);
        // Apply mish again
        ApplyMish(zLocal, xLocal, tmp1, tmp2);

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
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_mish_mish_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMishMish op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
