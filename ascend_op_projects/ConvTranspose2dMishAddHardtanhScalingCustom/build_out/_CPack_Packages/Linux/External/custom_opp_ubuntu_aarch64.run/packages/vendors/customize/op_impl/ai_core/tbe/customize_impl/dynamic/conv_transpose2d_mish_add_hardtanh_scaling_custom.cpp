
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMishAddHardtanhScale {
public:
    __aicore__ inline KernelMishAddHardtanhScale() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR addVal, GM_ADDR scaleVal, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // Read scalar add_value and scale from GM
        this->addValue = *((__gm__ float*)addVal);
        this->scaleValue = *((__gm__ float*)scaleVal);

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
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // Step 1: softplus = ln(1 + exp(x))
        AscendC::Exp(tmp1, xLocal, this->tileLength);          // tmp1 = exp(x)
        AscendC::Adds(tmp1, tmp1, (float)1.0, this->tileLength); // tmp1 = 1 + exp(x)
        AscendC::Ln(tmp1, tmp1, this->tileLength);              // tmp1 = ln(1 + exp(x)) = softplus

        // Step 2: tanh(softplus)
        // tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
        AscendC::Muls(tmp2, tmp1, (float)2.0, this->tileLength); // tmp2 = 2 * softplus
        AscendC::Exp(tmp2, tmp2, this->tileLength);               // tmp2 = exp(2*softplus)
        AscendC::Adds(tmp1, tmp2, (float)-1.0, this->tileLength); // tmp1 = exp(2*sp) - 1
        AscendC::Adds(tmp2, tmp2, (float)1.0, this->tileLength);  // tmp2 = exp(2*sp) + 1
        AscendC::Div(tmp1, tmp1, tmp2, this->tileLength);         // tmp1 = tanh(softplus)

        // Step 3: mish = x * tanh(softplus)
        AscendC::Mul(zLocal, xLocal, tmp1, this->tileLength);     // zLocal = mish(x)

        // Step 4: add value
        AscendC::Adds(zLocal, zLocal, this->addValue, this->tileLength); // zLocal = mish(x) + add_value

        // Step 5: hardtanh(x, -1, 1) = min(max(x, -1), 1)
        AscendC::Maxs(zLocal, zLocal, (float)-1.0, this->tileLength);
        AscendC::Mins(zLocal, zLocal, (float)1.0, this->tileLength);

        // Step 6: scale
        AscendC::Muls(zLocal, zLocal, this->scaleValue, this->tileLength);

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
    float addValue;
    float scaleValue;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transpose2d_mish_add_hardtanh_scaling_custom(GM_ADDR x, GM_ADDR addVal, GM_ADDR scaleVal, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMishAddHardtanhScale op;
    op.Init(x, addVal, scaleVal, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
