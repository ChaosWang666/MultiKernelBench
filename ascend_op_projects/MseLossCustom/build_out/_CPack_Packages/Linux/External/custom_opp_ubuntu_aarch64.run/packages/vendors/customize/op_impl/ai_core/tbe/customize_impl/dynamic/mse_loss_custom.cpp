
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMseLoss {
public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR result, uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        predGm.SetGlobalBuffer((__gm__ float *)predictions + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        targGm.SetGlobalBuffer((__gm__ float *)targets + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        resultGm.SetGlobalBuffer((__gm__ float *)result, 1);

        pipe.InitBuffer(inQueuePred, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueTarg, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(workBuf, 1, this->tileLength * sizeof(float));

        this->partialSum = 0.0f;
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        // Write the partial sum for this block, then do atomic-style reduction
        WriteResult();
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> predLocal = inQueuePred.AllocTensor<float>();
        AscendC::LocalTensor<float> targLocal = inQueueTarg.AllocTensor<float>();
        AscendC::DataCopy(predLocal, predGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(targLocal, targGm[progress * this->tileLength], this->tileLength);
        inQueuePred.EnQue(predLocal);
        inQueueTarg.EnQue(targLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> predLocal = inQueuePred.DeQue<float>();
        AscendC::LocalTensor<float> targLocal = inQueueTarg.DeQue<float>();
        AscendC::LocalTensor<float> diffLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> workLocal = workBuf.AllocTensor<float>();

        // diff = pred - targ
        AscendC::Sub(diffLocal, predLocal, targLocal, this->tileLength);
        // diff = diff * diff
        AscendC::Mul(diffLocal, diffLocal, diffLocal, this->tileLength);

        // Sum reduction of squared differences
        AscendC::ReduceSum(workLocal, diffLocal, workLocal, this->tileLength);
        float tileSum = workLocal.GetValue(0);
        this->partialSum += tileSum;

        outQueue.EnQue<float>(diffLocal);
        inQueuePred.FreeTensor(predLocal);
        inQueueTarg.FreeTensor(targLocal);
        workBuf.FreeTensor(workLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> diffLocal = outQueue.DeQue<float>();
        outQueue.FreeTensor(diffLocal);
    }
    __aicore__ inline void WriteResult()
    {
        // Divide partial sum by totalLength to get mean
        float meanVal = this->partialSum / (float)this->totalLength;

        // Use atomic add to accumulate across blocks
        AscendC::LocalTensor<float> workLocal = workBuf.AllocTensor<float>();
        workLocal.SetValue(0, meanVal);
        // Pad remaining with zeros for alignment
        for (int i = 1; i < 8; i++) {
            workLocal.SetValue(i, 0.0f);
        }

        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(resultGm[0], workLocal, 8);
        AscendC::SetAtomicNone();

        workBuf.FreeTensor(workLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueuePred, inQueueTarg;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targGm;
    AscendC::GlobalTensor<float> resultGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float partialSum;
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, result, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
