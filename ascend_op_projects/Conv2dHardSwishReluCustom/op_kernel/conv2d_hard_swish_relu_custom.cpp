
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelHardSwishRelu {
public:
    __aicore__ inline KernelHardSwishRelu() {}
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
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> t1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> t2 = tmpBuf2.Get<float>();

        // t1 = x + 3
        AscendC::Adds(t1, xLocal, (float)3.0f, this->tileLength);
        // clamp t1 to [0, 6]: t1 = min(max(t1, 0), 6)
        AscendC::Maxs(t1, t1, (float)0.0f, this->tileLength);
        AscendC::Mins(t1, t1, (float)6.0f, this->tileLength);
        // t1 = t1 / 6
        AscendC::Muls(t1, t1, (float)(1.0f / 6.0f), this->tileLength);
        // zLocal = x * t1  (this is hardswish)
        AscendC::Mul(zLocal, xLocal, t1, this->tileLength);
        // Apply ReLU: zLocal = max(zLocal, 0)
        AscendC::Maxs(zLocal, zLocal, (float)0.0f, this->tileLength);

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

extern "C" __global__ __aicore__ void conv2d_hard_swish_relu_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHardSwishRelu op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
