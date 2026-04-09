
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMinGptNewGelu {
public:
    __aicore__ inline KernelMinGptNewGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        
        // Temporary buffers for intermediate calculations
        AscendC::LocalTensor<float> tmp1 = pipe.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = pipe.AllocTensor<float>();

        // GELU approx: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // x^3
        AscendC::Mul(tmp1, xLocal, xLocal, this->tileLength); // x^2
        AscendC::Mul(tmp1, tmp1, xLocal, this->tileLength);   // x^3
        
        // 0.044715 * x^3
        AscendC::MulScalar(tmp1, tmp1, 0.044715f, this->tileLength);
        
        // x + 0.044715 * x^3
        AscendC::Add(tmp2, xLocal, tmp1, this->tileLength);
        
        // sqrt(2/pi) * (x + 0.044715 * x^3)
        AscendC::MulScalar(tmp2, tmp2, 0.79788456f, this->tileLength);
        
        // tanh(...)
        AscendC::Tanh(tmp2, tmp2, this->tileLength);
        
        // 1.0 + tanh(...)
        AscendC::AddScalar(tmp2, tmp2, 1.0f, this->tileLength);
        
        // 0.5 * x * (1.0 + tanh(...))
        AscendC::Mul(zLocal, xLocal, tmp2, this->tileLength);
        AscendC::MulScalar(zLocal, zLocal, 0.5f, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        pipe.FreeTensor(tmp1);
        pipe.FreeTensor(tmp2);
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
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void min_gpt_new_gelu_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinGptNewGelu op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
