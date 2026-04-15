
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelRmsprop {
public:
    __aicore__ inline KernelRmsprop() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR v, GM_ADDR paramOut,
                                 uint32_t totalLength, uint32_t tileNum, float lr, float alpha, float eps)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->lr = lr;
        this->alpha = alpha;
        this->oneMinusAlpha = 1.0f - alpha;
        this->eps = eps;

        paramGm.SetGlobalBuffer((__gm__ float *)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ float *)grad + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vGm.SetGlobalBuffer((__gm__ float *)v + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        paramOutGm.SetGlobalBuffer((__gm__ float *)paramOut + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.AllocTensor<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueV.EnQue(vLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // v_new = alpha * v + (1 - alpha) * grad^2
        // tmp1 = grad * grad
        AscendC::Mul(tmp1, gradLocal, gradLocal, this->tileLength);
        // tmp1 = (1-alpha) * grad^2
        AscendC::Muls(tmp1, tmp1, this->oneMinusAlpha, this->tileLength);
        // tmp2 = alpha * v
        AscendC::Muls(tmp2, vLocal, this->alpha, this->tileLength);
        // tmp1 = v_new = alpha*v + (1-alpha)*grad^2
        AscendC::Add(tmp1, tmp2, tmp1, this->tileLength);

        // param_out = param - lr * grad / (sqrt(v_new) + eps)
        // tmp2 = sqrt(v_new)
        AscendC::Sqrt(tmp2, tmp1, this->tileLength);
        // tmp2 = sqrt(v_new) + eps
        AscendC::Adds(tmp2, tmp2, this->eps, this->tileLength);
        // tmp1 = grad / (sqrt(v_new) + eps)
        AscendC::Div(tmp1, gradLocal, tmp2, this->tileLength);
        // tmp1 = lr * grad / (sqrt(v_new) + eps)
        AscendC::Muls(tmp1, tmp1, this->lr, this->tileLength);
        // outLocal = param - lr * grad / (sqrt(v_new) + eps)
        AscendC::Sub(outLocal, paramLocal, tmp1, this->tileLength);

        outQueue.EnQue(outLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueV.FreeTensor(vLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(paramOutGm[progress * this->tileLength], outLocal, this->tileLength);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueGrad, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> paramOutGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float lr;
    float alpha;
    float oneMinusAlpha;
    float eps;
};

extern "C" __global__ __aicore__ void rmsprop_custom(GM_ADDR param, GM_ADDR grad, GM_ADDR v, GM_ADDR paramOut, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsprop op;
    op.Init(param, grad, v, paramOut, tiling_data.totalLength, tiling_data.tileNum, tiling_data.lr, tiling_data.alpha, tiling_data.eps);
    op.Process();
}
