
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCrossEntropyLoss {
public:
    __aicore__ inline KernelCrossEntropyLoss() {}
    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR loss, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t numClasses)
    {
        this->batchSize = batchSize;
        this->numClasses = numClasses;
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockNum = AscendC::GetBlockNum();

        // Each block processes a subset of the batch
        this->samplesPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startSample = blockIdx * samplesPerBlock;
        this->endSample = startSample + samplesPerBlock;
        if (this->endSample > batchSize) {
            this->endSample = batchSize;
        }
        this->actualSamples = (endSample > startSample) ? (endSample - startSample) : 0;

        predGm.SetGlobalBuffer((__gm__ float*)predictions, batchSize * numClasses);
        targetGm.SetGlobalBuffer((__gm__ int32_t*)targets, batchSize);
        lossGm.SetGlobalBuffer((__gm__ float*)loss, 1);
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize);

        // Align numClasses to 8 for float (32 bytes)
        uint32_t alignedClasses = ((numClasses + 7) / 8) * 8;
        
        // We need buffers for one row of predictions
        pipe.InitBuffer(inQueuePred, BUFFER_NUM, alignedClasses * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedClasses * sizeof(float));
        // Small buffer for scalar results
        pipe.InitBuffer(workQueue, BUFFER_NUM, 8 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (actualSamples == 0) {
            // Write 0 to workspace for this block's region
            return;
        }

        float blockSum = 0.0f;

        for (uint32_t i = startSample; i < endSample; i++) {
            float sampleLoss = ProcessOneSample(i);
            blockSum += sampleLoss;
        }

        // Write partial sum to workspace
        // We need to align writes to 32 bytes
        AscendC::LocalTensor<float> wLocal = workQueue.AllocTensor<float>();
        wLocal.SetValue(0, blockSum);
        for (uint32_t j = 1; j < 8; j++) {
            wLocal.SetValue(j, 0.0f);
        }
        workQueue.EnQue(wLocal);
        wLocal = workQueue.DeQue<float>();
        // Write block result - each block writes to aligned position
        AscendC::DataCopy(workspaceGm[blockIdx * 8], wLocal, 8);
        workQueue.FreeTensor(wLocal);

        // Block 0 does final reduction
        AscendC::SetAtomicNone();
        if (blockIdx == 0) {
            // Sync - wait for all blocks
            pipe_barrier(PIPE_ALL);

            float totalSum = 0.0f;
            // Read all partial sums
            for (uint32_t b = 0; b < blockNum; b++) {
                AscendC::LocalTensor<float> rLocal = workQueue.AllocTensor<float>();
                AscendC::DataCopy(rLocal, workspaceGm[b * 8], 8);
                workQueue.EnQue(rLocal);
                rLocal = workQueue.DeQue<float>();
                totalSum += rLocal.GetValue(0);
                workQueue.FreeTensor(rLocal);
            }

            float meanLoss = totalSum / (float)batchSize;
            AscendC::LocalTensor<float> outLocal = workQueue.AllocTensor<float>();
            outLocal.SetValue(0, meanLoss);
            for (uint32_t j = 1; j < 8; j++) {
                outLocal.SetValue(j, 0.0f);
            }
            workQueue.EnQue(outLocal);
            outLocal = workQueue.DeQue<float>();
            AscendC::DataCopy(lossGm[0], outLocal, 8);
            workQueue.FreeTensor(outLocal);
        }
    }

private:
    __aicore__ inline float ProcessOneSample(uint32_t sampleIdx)
    {
        uint32_t alignedClasses = ((numClasses + 7) / 8) * 8;

        // Load predictions for this sample
        AscendC::LocalTensor<float> predLocal = inQueuePred.AllocTensor<float>();
        AscendC::DataCopy(predLocal, predGm[sampleIdx * numClasses], alignedClasses);
        inQueuePred.EnQue(predLocal);
        predLocal = inQueuePred.DeQue<float>();

        // Find max for numerical stability
        AscendC::LocalTensor<float> tmpLocal = outQueue.AllocTensor<float>();

        // ReduceMax
        float maxVal = predLocal.GetValue(0);
        for (uint32_t j = 1; j < numClasses; j++) {
            float v = predLocal.GetValue(j);
            if (v > maxVal) maxVal = v;
        }

        // Subtract max: predLocal = predLocal - maxVal
        AscendC::Adds(tmpLocal, predLocal, -maxVal, alignedClasses);

        // Exp
        AscendC::Exp(tmpLocal, tmpLocal, alignedClasses);

        // Sum of exp
        float sumExp = 0.0f;
        for (uint32_t j = 0; j < numClasses; j++) {
            sumExp += tmpLocal.GetValue(j);
        }

        // Read target
        // We need to read target - but int32 needs aligned read
        // Read 8 int32 values aligned
        int32_t targetVal;
        // Use scalar read approach
        // For targets, we do an aligned copy
        AscendC::LocalTensor<float> targetBuf = outQueue.AllocTensor<float>();
        uint32_t alignedTargetOffset = (sampleIdx / 8) * 8;
        AscendC::DataCopy(targetBuf, (__gm__ float*)((__gm__ int32_t*)targetGm.GetPhyAddr() + alignedTargetOffset), 8);
        outQueue.EnQue(targetBuf);
        targetBuf = outQueue.DeQue<float>();
        // Reinterpret
        int32_t* targetPtr = (int32_t*)&targetBuf;
        targetVal = targetPtr[sampleIdx - alignedTargetOffset];
        outQueue.FreeTensor(targetBuf);

        // cross entropy = -log(exp(x_target - max) / sum_exp) = -(x_target - max) + log(sum_exp)
        float logSumExp = AscendC::log(sumExp);
        float targetPredShifted = predLocal.GetValue(targetVal) - maxVal;
        float loss = -targetPredShifted + logSumExp;

        outQueue.FreeTensor(tmpLocal);
        inQueuePred.FreeTensor(predLocal);

        return loss;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueuePred;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> workQueue;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<int32_t> targetGm;
    AscendC::GlobalTensor<float> lossGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint32_t batchSize;
    uint32_t numClasses;
    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t samplesPerBlock;
    uint32_t startSample;
    uint32_t endSample;
    uint32_t actualSamples;
};

extern "C" __global__ __aicore__ void cross_entropy_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossEntropyLoss op;
    op.Init(predictions, targets, loss, workspace, tiling_data.batchSize, tiling_data.numClasses);
    op.Process();
}
