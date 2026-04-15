
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmSigmoidScalingResidualAdd {
public:
    __aicore__ inline KernelGemmSigmoidScalingResidualAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, float scalingFactor)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->scalingFactor = scalingFactor;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
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

        // sigmoid(x) = 1 / (1 + exp(-x))
        // Negate x
        AscendC::Muls(tmpLocal, xLocal, (float)(-1.0f), this->tileLength);
        // exp(-x)
        AscendC::Exp(tmpLocal, tmpLocal, this->tileLength);
        // 1 + exp(-x)
        AscendC::Adds(tmpLocal, tmpLocal, (float)(1.0f), this->tileLength);
        // 1 / (1 + exp(-x))
        AscendC::Reciprocal(tmpLocal, tmpLocal, this->tileLength);

        // sigmoid(x) * scaling_factor
        AscendC::Muls(tmpLocal, tmpLocal, this->scalingFactor, this->tileLength);
        // sigmoid(x) * scaling_factor + x (residual add)
        AscendC::Add(zLocal, tmpLocal, xLocal, this->tileLength);

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
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void gemm_sigmoid_scaling_residual_add_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmSigmoidScalingResidualAdd op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.scalingFactor);
    op.Process();
}
