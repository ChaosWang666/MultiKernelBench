
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelKlDivLoss {
public:
    __aicore__ inline KernelKlDivLoss() {}
    __aicore__ inline void Init(GM_ADDR log_predictions, GM_ADDR targets, GM_ADDR loss,
                                 uint32_t totalLength, uint32_t tileNum, uint32_t batchSize)
    {
        this->totalLength = totalLength;
        this->batchSize = batchSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        logPredGm.SetGlobalBuffer((__gm__ float *)log_predictions + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        targetsGm.SetGlobalBuffer((__gm__ float *)targets + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        lossGm.SetGlobalBuffer((__gm__ float *)loss, 1);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
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
        // Write partial sum for this block to lossGm using atomic add
        // We'll use block 0 to aggregate; each block atomically adds
        // For simplicity, write partial sums to the output and let host handle,
        // or use AtomicAdd. We'll do a simple approach: write to lossGm[blockIdx]
        // Actually, lossGm is only 1 element. Let's use a different strategy.
        // We accumulate partialSum per block and use AtomicAdd to lossGm[0].
        
        // Use pipe to write partial sum
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        tmpLocal.SetValue(0, this->partialSum);
        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(lossGm[0], tmpLocal, 1);
        AscendC::SetAtomicNone();
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, logPredGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, targetsGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();

        // KL divergence: targets * (log(targets) - log_predictions)
        // = targets * log(targets) - targets * log_predictions
        // Using the PyTorch convention for kl_div with log input:
        // loss = exp(log_predictions) - targets * log_predictions  ... No.
        // Actually torch.nn.functional.kl_div(log_input, target, reduction='batchmean')
        // = sum(target * (log(target) - log_input)) / batch_size
        // But PyTorch kl_div actually computes: target * (log(target) - input) where input = log(predictions)
        // The formula is: F.kl_div(input, target) = target * (log(target) - input)
        // So element-wise: target_i * log(target_i) - target_i * input_i
        
        // Compute: targets * log(targets) - targets * log_predictions
        // zLocal = targets * log(targets) - targets * log_predictions

        // tmp = log(targets)
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::Ln(tmpLocal, yLocal, this->tileLength);
        // tmp = targets * log(targets)
        AscendC::Mul(tmpLocal, yLocal, tmpLocal, this->tileLength);
        // zLocal = targets * log_predictions
        AscendC::Mul(zLocal, yLocal, xLocal, this->tileLength);
        // zLocal = targets * log(targets) - targets * log_predictions
        AscendC::Sub(zLocal, tmpLocal, zLocal, this->tileLength);

        // Accumulate sum
        float tileSum = 0.0f;
        for (uint32_t i = 0; i < this->tileLength; i++) {
            tileSum += zLocal.GetValue(i);
        }
        this->partialSum += tileSum;

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> logPredGm;
    AscendC::GlobalTensor<float> targetsGm;
    AscendC::GlobalTensor<float> lossGm;
    uint32_t totalLength;
    uint32_t batchSize;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float partialSum;
};

extern "C" __global__ __aicore__ void kl_div_loss_custom(GM_ADDR log_predictions, GM_ADDR targets, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelKlDivLoss op;
    op.Init(log_predictions, targets, loss, tiling_data.totalLength, tiling_data.tileNum, tiling_data.batchSize);
    op.Process();
}
