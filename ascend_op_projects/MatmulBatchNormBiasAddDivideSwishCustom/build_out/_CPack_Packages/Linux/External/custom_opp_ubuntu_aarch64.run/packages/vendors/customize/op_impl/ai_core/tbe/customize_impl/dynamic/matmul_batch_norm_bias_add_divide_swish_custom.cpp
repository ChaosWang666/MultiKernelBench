
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelBiasAddDivideSwish {
public:
    __aicore__ inline KernelBiasAddDivideSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR divideVal, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, 1);
        divideValGm.SetGlobalBuffer((__gm__ float *)divideVal, 1);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuffer, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();

        // bias add: x = x + bias
        float biasVal = biasGm.GetValue(0);
        AscendC::Adds(zLocal, xLocal, biasVal, this->tileLength);

        // divide: x = x / divide_value
        float divVal = divideValGm.GetValue(0);
        float invDiv = 1.0f / divVal;
        AscendC::Muls(zLocal, zLocal, invDiv, this->tileLength);

        // swish: x = x * sigmoid(x)
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Compute sigmoid into tmpLocal
        AscendC::Muls(tmpLocal, zLocal, -1.0f, this->tileLength);
        AscendC::Exp(tmpLocal, tmpLocal, this->tileLength);
        AscendC::Adds(tmpLocal, tmpLocal, 1.0f, this->tileLength);
        AscendC::Reciprocal(tmpLocal, tmpLocal, this->tileLength);

        // x * sigmoid(x)
        AscendC::Mul(zLocal, zLocal, tmpLocal, this->tileLength);

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> divideValGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_batch_norm_bias_add_divide_swish_custom(GM_ADDR x, GM_ADDR bias, GM_ADDR divide_val, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBiasAddDivideSwish op;
    op.Init(x, bias, divide_val, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
