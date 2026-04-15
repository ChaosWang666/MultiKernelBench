
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLamb {
public:
    __aicore__ inline KernelLamb() {}
    __aicore__ inline void Init(GM_ADDR param, GM_ADDR m, GM_ADDR v, GM_ADDR out,
                                GM_ADDR workspace, uint32_t totalLength, uint32_t tileNum,
                                float lr, float eps)
    {
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->lr = lr;
        this->eps = eps;

        paramGm.SetGlobalBuffer((__gm__ float *)param + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        mGm.SetGlobalBuffer((__gm__ float *)m + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        vGm.SetGlobalBuffer((__gm__ float *)v + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float *)out + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace, AscendC::GetBlockNum() * 2);

        pipe.InitBuffer(inQueueParam, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueM, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Pass 1: compute r = m / (sqrt(v) + eps), accumulate paramNormSq and rNormSq
        float paramNormSqLocal = 0.0f;
        float rNormSqLocal = 0.0f;

        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            // CopyIn param, m, v
            AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
            AscendC::LocalTensor<float> mLocal = inQueueM.AllocTensor<float>();
            AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();

            AscendC::DataCopy(paramLocal, paramGm[i * this->tileLength], this->tileLength);
            AscendC::DataCopy(mLocal, mGm[i * this->tileLength], this->tileLength);
            AscendC::DataCopy(vLocal, vGm[i * this->tileLength], this->tileLength);

            inQueueParam.EnQue(paramLocal);
            inQueueM.EnQue(mLocal);
            inQueueV.EnQue(vLocal);

            // Compute
            paramLocal = inQueueParam.DeQue<float>();
            mLocal = inQueueM.DeQue<float>();
            vLocal = inQueueV.DeQue<float>();
            AscendC::LocalTensor<float> rLocal = outQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

            // sqrt(v)
            AscendC::Sqrt(rLocal, vLocal, this->tileLength);
            // sqrt(v) + eps
            AscendC::Adds(rLocal, rLocal, this->eps, this->tileLength);
            // r = m / (sqrt(v) + eps)
            AscendC::Div(rLocal, mLocal, rLocal, this->tileLength);

            // paramNormSq += sum(param^2)
            AscendC::Mul(tmpLocal, paramLocal, paramLocal, this->tileLength);
            float sumP = 0.0f;
            AscendC::ReduceSum(tmpLocal, tmpLocal, tmpLocal, this->tileLength);
            sumP = tmpLocal.GetValue(0);
            paramNormSqLocal += sumP;

            // rNormSq += sum(r^2)
            AscendC::Mul(tmpLocal, rLocal, rLocal, this->tileLength);
            float sumR = 0.0f;
            AscendC::ReduceSum(tmpLocal, tmpLocal, tmpLocal, this->tileLength);
            sumR = tmpLocal.GetValue(0);
            rNormSqLocal += sumR;

            // Store r temporarily in outGm
            outQueue.EnQue(rLocal);
            rLocal = outQueue.DeQue<float>();
            AscendC::DataCopy(outGm[i * this->tileLength], rLocal, this->tileLength);
            outQueue.FreeTensor(rLocal);

            inQueueParam.FreeTensor(paramLocal);
            inQueueM.FreeTensor(mLocal);
            inQueueV.FreeTensor(vLocal);
        }

        // Write partial sums to workspace
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // Use a small tensor to write
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        tmpLocal.SetValue(0, paramNormSqLocal);
        AscendC::DataCopy(workspaceGm[blockIdx * 2], tmpLocal, 1);
        tmpLocal.SetValue(0, rNormSqLocal);
        AscendC::DataCopy(workspaceGm[blockIdx * 2 + 1], tmpLocal, 1);

        // Synchronize all blocks
        AscendC::SyncAll();

        // Each block reads all partial sums and computes global norms
        float totalParamNormSq = 0.0f;
        float totalRNormSq = 0.0f;
        uint32_t numBlocks = AscendC::GetBlockNum();
        for (uint32_t b = 0; b < numBlocks; b++) {
            AscendC::DataCopy(tmpLocal, workspaceGm[b * 2], 8);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            totalParamNormSq += tmpLocal.GetValue(0);
            totalRNormSq += tmpLocal.GetValue(1);
        }

        float paramNorm = sqrtf(totalParamNormSq);
        float rNorm = sqrtf(totalRNormSq);
        float trustRatio = paramNorm / (rNorm + this->eps);
        float scale = this->lr * trustRatio;

        // Pass 2: param = param - scale * r
        for (int32_t i = 0; i < loopCount; i++) {
            AscendC::LocalTensor<float> paramLocal = inQueueParam.AllocTensor<float>();
            AscendC::LocalTensor<float> rLocal = inQueueM.AllocTensor<float>();

            AscendC::DataCopy(paramLocal, paramGm[i * this->tileLength], this->tileLength);
            AscendC::DataCopy(rLocal, outGm[i * this->tileLength], this->tileLength);

            inQueueParam.EnQue(paramLocal);
            inQueueM.EnQue(rLocal);

            paramLocal = inQueueParam.DeQue<float>();
            rLocal = inQueueM.DeQue<float>();
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

            // scale * r
            AscendC::Muls(outLocal, rLocal, scale, this->tileLength);
            // param - scale * r
            AscendC::Sub(outLocal, paramLocal, outLocal, this->tileLength);

            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<float>();
            AscendC::DataCopy(outGm[i * this->tileLength], outLocal, this->tileLength);
            outQueue.FreeTensor(outLocal);

            inQueueParam.FreeTensor(paramLocal);
            inQueueM.FreeTensor(rLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueParam, inQueueM, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> paramGm, mGm, vGm, outGm, workspaceGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float lr;
    float eps;
};

extern "C" __global__ __aicore__ void lamb_custom(GM_ADDR param, GM_ADDR m, GM_ADDR v, GM_ADDR out,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLamb op;
    op.Init(param, m, v, out, workspace, tiling_data.totalLength, tiling_data.tileNum,
            tiling_data.lr, tiling_data.eps);
    op.Process();
}
