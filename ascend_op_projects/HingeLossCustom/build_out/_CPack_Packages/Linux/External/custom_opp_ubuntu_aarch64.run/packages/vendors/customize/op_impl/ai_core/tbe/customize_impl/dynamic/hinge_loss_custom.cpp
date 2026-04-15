
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelHingeLoss {
public:
    __aicore__ inline KernelHingeLoss() {}
    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR output, uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        predGm.SetGlobalBuffer((__gm__ float *)predictions + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        targGm.SetGlobalBuffer((__gm__ float *)targets + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float *)output, 1);

        pipe.InitBuffer(inQueuePred, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueTarg, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, this->tileLength * sizeof(float));

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
        // Write final reduced result
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
        AscendC::LocalTensor<float> resultLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        // Compute predictions * targets
        AscendC::Mul(resultLocal, predLocal, targLocal, this->tileLength);

        // Compute 1 - predictions * targets
        AscendC::Muls(resultLocal, resultLocal, (float)-1.0f, this->tileLength);
        AscendC::Adds(resultLocal, resultLocal, (float)1.0f, this->tileLength);

        // clamp(x, min=0) = max(x, 0)
        AscendC::Maxs(resultLocal, resultLocal, (float)0.0f, this->tileLength);

        // Accumulate sum for mean computation
        // ReduceSum: sum all elements in the tile
        AscendC::ReduceSum(tmpLocal, resultLocal, tmpLocal, this->tileLength);
        this->partialSum += tmpLocal.GetValue(0);

        outQueue.EnQue<float>(resultLocal);
        inQueuePred.FreeTensor(predLocal);
        inQueueTarg.FreeTensor(targLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> resultLocal = outQueue.DeQue<float>();
        outQueue.FreeTensor(resultLocal);
    }
    __aicore__ inline void WriteResult()
    {
        // Use atomic add to accumulate across blocks
        // Each block writes its partial sum, then block 0 divides by totalLength
        // We use a simple approach: write partial sum to a temp location per block
        // Actually, we need to use AtomicAdd for cross-block reduction.
        // Simple approach: use AscendC::AtomicAdd
        float mean = this->partialSum / (float)this->totalLength;
        
        AscendC::LocalTensor<float> resultLocal = outQueue.AllocTensor<float>();
        resultLocal.SetValue(0, mean);
        // Use DataCopy with atomic add
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(outGm[0], resultLocal, 1);
        AscendC::SetAtomicNone();
        outQueue.FreeTensor(resultLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueuePred, inQueueTarg;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targGm;
    AscendC::GlobalTensor<float> outGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float partialSum;
};

extern "C" __global__ __aicore__ void hinge_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHingeLoss op;
    op.Init(predictions, targets, output, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
