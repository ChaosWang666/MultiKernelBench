
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAdagrad {
public:
    __aicore__ inline KernelAdagrad() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR accum, GM_ADDR paramOut,
                                 uint32_t totalLength, uint32_t tileNum, float lr, float eps)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->lr = lr;
        this->eps = eps;

        paramGm.SetGlobalBuffer((__gm__ float *)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        gradGm.SetGlobalBuffer((__gm__ float *)grad + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        accumGm.SetGlobalBuffer((__gm__ float *)accum + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        paramOutGm.SetGlobalBuffer((__gm__ float *)paramOut + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueGrad, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueAccum, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> accumLocal = inQueueAccum.AllocTensor<float>();
        AscendC::DataCopy(paramLocal, paramGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(gradLocal, gradGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(accumLocal, accumGm[progress * this->tileLength], this->tileLength);
        inQueueParam.EnQue(paramLocal);
        inQueueGrad.EnQue(gradLocal);
        inQueueAccum.EnQue(accumLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> paramLocal = inQueueParam.DeQue<float>();
        AscendC::LocalTensor<float> gradLocal = inQueueGrad.DeQue<float>();
        AscendC::LocalTensor<float> accumLocal = inQueueAccum.DeQue<float>();
        AscendC::LocalTensor<float> paramOutLocal = outQueueParam.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

        // accum = accum + grad * grad
        AscendC::Mul(tmp1, gradLocal, gradLocal, this->tileLength);
        AscendC::Add(accumLocal, accumLocal, tmp1, this->tileLength);

        // sqrt(accum)
        AscendC::Sqrt(tmp1, accumLocal, this->tileLength);

        // sqrt(accum) + eps
        AscendC::Adds(tmp1, tmp1, this->eps, this->tileLength);

        // lr * grad
        AscendC::Muls(tmp2, gradLocal, this->lr, this->tileLength);

        // lr * grad / (sqrt(accum) + eps)
        AscendC::Div(tmp1, tmp2, tmp1, this->tileLength);

        // param - lr * grad / (sqrt(accum) + eps)
        AscendC::Sub(paramOutLocal, paramLocal, tmp1, this->tileLength);

        outQueueParam.EnQue<float>(paramOutLocal);
        inQueueParam.FreeTensor(paramLocal);
        inQueueGrad.FreeTensor(gradLocal);
        inQueueAccum.FreeTensor(accumLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> paramOutLocal = outQueueParam.DeQue<float>();
        AscendC::DataCopy(paramOutGm[progress * this->tileLength], paramOutLocal, this->tileLength);
        outQueueParam.FreeTensor(paramOutLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueGrad, inQueueAccum;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueParam;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> paramGm;
    AscendC::GlobalTensor<float> gradGm;
    AscendC::GlobalTensor<float> accumGm;
    AscendC::GlobalTensor<float> paramOutGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float lr;
    float eps;
};

extern "C" __global__ __aicore__ void adagrad_custom(GM_ADDR param, GM_ADDR grad, GM_ADDR accum, GM_ADDR paramOut, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdagrad op;
    op.Init(param, grad, accum, paramOut, tiling_data.totalLength, tiling_data.tileNum, tiling_data.lr, tiling_data.eps);
    op.Process();
}
