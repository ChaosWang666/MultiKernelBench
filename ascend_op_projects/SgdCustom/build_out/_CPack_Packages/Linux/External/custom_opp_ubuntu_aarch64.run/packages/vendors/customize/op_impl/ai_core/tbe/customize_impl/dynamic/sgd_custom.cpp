
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelSgd {
public:
    __aicore__ inline KernelSgd() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR velocity, GM_ADDR paramOut,
                                uint32_t totalLength, uint32_t tileNum, float momentum, float lr)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->momentum = momentum;
        this->lr = lr;

        paramGm.SetGlobalBuffer((__gm__ float *)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ float *)grad + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        velocityGm.SetGlobalBuffer((__gm__ float *)velocity + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        paramOutGm.SetGlobalBuffer((__gm__ float *)paramOut + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueVelocity, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueParamOut, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.AllocTensor<float>();
        AscendC::LocalTensor<float> velocityLocal = inQueueVelocity.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(velocityLocal, velocityGm[progress * this->tileLength], this->tileLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueVelocity.EnQue(velocityLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> velocityLocal = inQueueVelocity.DeQue<float>();
        AscendC::LocalTensor<float> paramOutLocal = outQueueParamOut.AllocTensor<float>();

        // velocity_new = momentum * velocity + grad
        AscendC::Muls(velocityLocal, velocityLocal, this->momentum, this->tileLength);
        AscendC::Add(velocityLocal, velocityLocal, gradLocal, this->tileLength);

        // param_out = param - lr * velocity_new
        AscendC::Muls(paramOutLocal, velocityLocal, this->lr, this->tileLength);
        AscendC::Sub(paramOutLocal, paramLocal, paramOutLocal, this->tileLength);

        outQueueParamOut.EnQue<float>(paramOutLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueVelocity.FreeTensor(velocityLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> paramOutLocal = outQueueParamOut.DeQue<float>();
        AscendC::DataCopy(paramOutGm[progress * this->tileLength], paramOutLocal, this->tileLength);
        outQueueParamOut.FreeTensor(paramOutLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueGrad, inQueueVelocity;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueParamOut;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> velocityGm;
    AscendC::GlobalTensor<float> paramOutGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float momentum;
    float lr;
};

extern "C" __global__ __aicore__ void sgd_custom(GM_ADDR param, GM_ADDR grad, GM_ADDR velocity, GM_ADDR param_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSgd op;
    op.Init(param, grad, velocity, param_out, tiling_data.totalLength, tiling_data.tileNum, tiling_data.momentum, tiling_data.lr);
    op.Process();
}
