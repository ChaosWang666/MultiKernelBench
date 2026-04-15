
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAdam {
public:
    __aicore__ inline KernelAdam() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR paramOut,
                                 uint32_t totalLength, uint32_t tileNum,
                                 float beta1, float beta2, float lr, float eps,
                                 float beta1CorrInv, float beta2CorrInv)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->lr = lr;
        this->eps = eps;
        this->beta1CorrInv = beta1CorrInv;
        this->beta2CorrInv = beta2CorrInv;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        paramGm.SetGlobalBuffer((__gm__ float *)param + offset, this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ float *)grad + offset, this->blockLength);
        mGm.SetGlobalBuffer((__gm__ float *)m + offset, this->blockLength);
        vGm.SetGlobalBuffer((__gm__ float *)v + offset, this->blockLength);
        paramOutGm.SetGlobalBuffer((__gm__ float *)paramOut + offset, this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueM, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> mLocal = inQueueM.AllocTensor<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(mLocal, mGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(vLocal, vGm[progress * this->tileLength], this->tileLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueM.EnQue(mLocal);
        inQueueV.EnQue(vLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> mLocal = inQueueM.DeQue<float>();
        AscendC::LocalTensor<float> vLocal = inQueueV.DeQue<float>();
        AscendC::LocalTensor<float> resultLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // m_new = beta1 * m + (1 - beta1) * grad
        AscendC::Muls(tmp1, mLocal, this->beta1, this->tileLength);
        AscendC::Muls(tmp2, gradLocal, 1.0f - this->beta1, this->tileLength);
        AscendC::Add(mLocal, tmp1, tmp2, this->tileLength);

        // v_new = beta2 * v + (1 - beta2) * grad^2
        AscendC::Muls(tmp1, vLocal, this->beta2, this->tileLength);
        AscendC::Mul(tmp2, gradLocal, gradLocal, this->tileLength);
        AscendC::Muls(tmp2, tmp2, 1.0f - this->beta2, this->tileLength);
        AscendC::Add(vLocal, tmp1, tmp2, this->tileLength);

        // m_hat = m_new * beta1CorrInv
        AscendC::Muls(tmp1, mLocal, this->beta1CorrInv, this->tileLength);

        // v_hat = v_new * beta2CorrInv
        AscendC::Muls(tmp2, vLocal, this->beta2CorrInv, this->tileLength);

        // sqrt(v_hat) + eps
        AscendC::Sqrt(tmp2, tmp2, this->tileLength);
        AscendC::Adds(tmp2, tmp2, this->eps, this->tileLength);

        // lr * m_hat / (sqrt(v_hat) + eps)
        AscendC::Div(tmp1, tmp1, tmp2, this->tileLength);
        AscendC::Muls(tmp1, tmp1, this->lr, this->tileLength);

        // param_new = param - update
        AscendC::Sub(resultLocal, paramLocal, tmp1, this->tileLength);

        outQueue.EnQue(resultLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueM.FreeTensor(mLocal);
        inQueueV.FreeTensor(vLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> resultLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(paramOutGm[progress * this->tileLength], resultLocal, this->tileLength);
        outQueue.FreeTensor(resultLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueGrad, inQueueM, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> mGm;
    AscendC::GlobalTensor<float> vGm;
    AscendC::GlobalTensor<float> paramOutGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float beta1;
    float beta2;
    float lr;
    float eps;
    float beta1CorrInv;
    float beta2CorrInv;
};

extern "C" __global__ __aicore__ void adam_custom(GM_ADDR param, GM_ADDR grad, GM_ADDR m, GM_ADDR v,
                                                    GM_ADDR param_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdam op;
    op.Init(param, grad, m, v, param_out,
            tiling_data.totalLength, tiling_data.tileNum,
            tiling_data.beta1, tiling_data.beta2, tiling_data.lr, tiling_data.eps,
            tiling_data.beta1CorrInv, tiling_data.beta2CorrInv);
    op.Process();
}
