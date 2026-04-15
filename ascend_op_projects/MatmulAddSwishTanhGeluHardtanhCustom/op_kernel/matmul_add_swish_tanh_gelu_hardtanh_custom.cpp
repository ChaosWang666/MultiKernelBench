
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulAddSwishTanhGeluHardtanh {
public:
    __aicore__ inline KernelMatmulAddSwishTanhGeluHardtanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR addVal, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, uint32_t addValueLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->addValueLength = addValueLength;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        addValGm.SetGlobalBuffer((__gm__ float *)addVal, addValueLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueAdd, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> addLocal = inQueueAdd.AllocTensor<float>();
        uint32_t offset = progress * this->tileLength;
        AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
        // add_value is broadcast along rows; compute the column offset
        // The global offset for this block is blockLength * blockIdx + progress * tileLength
        // Column index = globalOffset % addValueLength
        uint32_t globalOffset = this->blockLength * AscendC::GetBlockIdx() + offset;
        uint32_t colOffset = globalOffset % this->addValueLength;
        AscendC::DataCopy(addLocal, addValGm[colOffset], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueAdd.EnQue(addLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> addLocal = inQueueAdd.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // x = x + add_value
        AscendC::Add(xLocal, xLocal, addLocal, this->tileLength);

        // Swish: x = sigmoid(x) * x
        // sigmoid(x) = 1/(1+exp(-x))
        // tmp1 = -x
        AscendC::Muls(tmp1, xLocal, (float)-1.0f, this->tileLength);
        // tmp1 = exp(-x)
        AscendC::Exp(tmp1, tmp1, this->tileLength);
        // tmp1 = 1 + exp(-x)
        AscendC::Adds(tmp1, tmp1, (float)1.0f, this->tileLength);
        // tmp1 = 1 / (1 + exp(-x))
        AscendC::Reciprocal(tmp1, tmp1, this->tileLength);
        // xLocal = sigmoid(x) * x
        AscendC::Mul(xLocal, tmp1, xLocal, this->tileLength);

        // Tanh: x = tanh(x)
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // Use: tanh(x) = 2*sigmoid(2x) - 1
        AscendC::Muls(tmp1, xLocal, (float)2.0f, this->tileLength);
        // tmp1 = -2x
        AscendC::Muls(tmp1, tmp1, (float)-1.0f, this->tileLength);
        // tmp1 = exp(-2x)
        AscendC::Exp(tmp1, tmp1, this->tileLength);
        // tmp1 = 1 + exp(-2x)
        AscendC::Adds(tmp1, tmp1, (float)1.0f, this->tileLength);
        // tmp1 = 1/(1+exp(-2x)) = sigmoid(2x)
        AscendC::Reciprocal(tmp1, tmp1, this->tileLength);
        // tmp1 = 2*sigmoid(2x)
        AscendC::Muls(tmp1, tmp1, (float)2.0f, this->tileLength);
        // xLocal = 2*sigmoid(2x) - 1 = tanh(x)
        AscendC::Adds(xLocal, tmp1, (float)-1.0f, this->tileLength);

        // GELU: x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // tmp1 = x^2
        AscendC::Mul(tmp1, xLocal, xLocal, this->tileLength);
        // tmp1 = x^3
        AscendC::Mul(tmp1, tmp1, xLocal, this->tileLength);
        // tmp1 = 0.044715 * x^3
        AscendC::Muls(tmp1, tmp1, (float)0.044715f, this->tileLength);
        // tmp1 = x + 0.044715 * x^3
        AscendC::Add(tmp1, tmp1, xLocal, this->tileLength);
        // tmp1 = sqrt(2/pi) * (x + 0.044715 * x^3), sqrt(2/pi) ~ 0.7978845608
        AscendC::Muls(tmp1, tmp1, (float)0.7978845608f, this->tileLength);
        // tanh(tmp1): use 2*sigmoid(2*tmp1) - 1
        AscendC::Muls(tmp1, tmp1, (float)2.0f, this->tileLength);
        AscendC::Muls(tmp1, tmp1, (float)-1.0f, this->tileLength);
        AscendC::Exp(tmp1, tmp1, this->tileLength);
        AscendC::Adds(tmp1, tmp1, (float)1.0f, this->tileLength);
        AscendC::Reciprocal(tmp1, tmp1, this->tileLength);
        AscendC::Muls(tmp1, tmp1, (float)2.0f, this->tileLength);
        AscendC::Adds(tmp1, tmp1, (float)-1.0f, this->tileLength);
        // tmp1 = 1 + tanh(...)
        AscendC::Adds(tmp1, tmp1, (float)1.0f, this->tileLength);
        // tmp1 = 0.5 * x * (1 + tanh(...))
        AscendC::Mul(tmp1, tmp1, xLocal, this->tileLength);
        AscendC::Muls(xLocal, tmp1, (float)0.5f, this->tileLength);

        // Hardtanh: clamp to [-1, 1]
        // min with 1.0
        AscendC::Mins(xLocal, xLocal, (float)1.0f, this->tileLength);
        // max with -1.0
        AscendC::Maxs(xLocal, xLocal, (float)-1.0f, this->tileLength);

        AscendC::DataCopy(zLocal, xLocal, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueAdd.FreeTensor(addLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueAdd;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> addValGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t addValueLength;
};

extern "C" __global__ __aicore__ void matmul_add_swish_tanh_gelu_hardtanh_custom(GM_ADDR x, GM_ADDR add_value, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulAddSwishTanhGeluHardtanh op;
    op.Init(x, add_value, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.addValueLength);
    op.Process();
}
