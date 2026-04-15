
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSubtractMish {
public:
    __aicore__ inline KernelSubtractMish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, float subtractValue)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->subtractValue = subtractValue;

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
        AscendC::LocalTensor<float> temp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> temp2 = tmpBuf2.Get<float>();

        // x = x - subtractValue (combined subtract_value_1 + subtract_value_2)
        AscendC::Adds(xLocal, xLocal, -this->subtractValue, this->tileLength);

        // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        // Step 1: compute exp(x)
        AscendC::Exp(temp1, xLocal, this->tileLength);
        // Step 2: compute 1 + exp(x)
        AscendC::Adds(temp1, temp1, 1.0f, this->tileLength);
        // Step 3: compute ln(1 + exp(x)) = softplus(x)
        AscendC::Ln(temp1, temp1, this->tileLength);
        // Step 4: compute tanh(softplus(x))
        AscendC::Tanh(temp2, temp1, this->tileLength);
        // Step 5: compute x * tanh(softplus(x))
        AscendC::Mul(zLocal, xLocal, temp2, this->tileLength);

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
    float subtractValue;
};

extern "C" __global__ __aicore__ void conv2d_subtract_subtract_mish_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubtractMish op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.subtractValue);
    op.Process();
}
