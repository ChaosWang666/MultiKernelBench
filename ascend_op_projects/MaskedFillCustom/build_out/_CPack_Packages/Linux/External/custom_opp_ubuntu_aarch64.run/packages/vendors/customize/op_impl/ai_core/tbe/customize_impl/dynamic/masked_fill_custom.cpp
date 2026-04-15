
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMaskedFill {
public:
    __aicore__ inline KernelMaskedFill() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        maskGm.SetGlobalBuffer((__gm__ int8_t *)mask + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, this->tileLength * sizeof(int8_t));
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
        AscendC::LocalTensor<int8_t> maskLocal = inQueueMask.AllocTensor<int8_t>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(maskLocal, maskGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueMask.EnQue(maskLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<int8_t> maskLocal = inQueueMask.DeQue<int8_t>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> maskFloat = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> fillVal = tmpBuf2.Get<float>();

        // Cast mask (int8) to float: 0 -> 0.0, 1 -> 1.0
        AscendC::Cast(maskFloat, maskLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);

        // fillVal = maskFloat * (-inf)
        // We use a large negative number representation for -inf
        // neg_inf float = 0xFF800000
        float negInf;
        *(uint32_t*)(&negInf) = 0xFF800000;
        AscendC::Muls(fillVal, maskFloat, negInf, this->tileLength);

        // invertedMask = 1.0 - maskFloat
        AscendC::Muls(maskFloat, maskFloat, (float)-1.0, this->tileLength);
        AscendC::Adds(maskFloat, maskFloat, (float)1.0, this->tileLength);

        // zLocal = xLocal * invertedMask + fillVal
        AscendC::Mul(zLocal, xLocal, maskFloat, this->tileLength);
        AscendC::Add(zLocal, zLocal, fillVal, this->tileLength);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueMask.FreeTensor(maskLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueMask;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int8_t> maskGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void masked_fill_custom(GM_ADDR x, GM_ADDR mask, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedFill op;
    op.Init(x, mask, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
