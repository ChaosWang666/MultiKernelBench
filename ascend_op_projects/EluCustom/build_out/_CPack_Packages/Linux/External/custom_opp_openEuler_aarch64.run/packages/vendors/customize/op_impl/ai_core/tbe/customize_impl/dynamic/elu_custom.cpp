
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelElu {
public:
    __aicore__ inline KernelElu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR alpha, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        alphaGm.SetGlobalBuffer((__gm__ float *)alpha, 1);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        float alphaVal = alphaGm.GetValue(0);
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i, alphaVal);
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
    __aicore__ inline void Compute(int32_t progress, float alphaVal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        
        AscendC::LocalTensor<float> mask = xLocal < 0.0f;
        AscendC::LocalTensor<float> temp = xLocal;
        
        AscendC::Exp(temp, xLocal, this->tileLength);
        AscendC::Sub(temp, temp, 1.0f, this->tileLength);
        AscendC::Mul(temp, temp, alphaVal, this->tileLength);
        
        AscendC::Select(zLocal, mask, temp, xLocal, this->tileLength);
        
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
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> alphaGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void elu_custom(GM_ADDR x, GM_ADDR alpha, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelElu op;
    op.Init(x, alpha, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
