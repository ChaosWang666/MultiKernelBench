
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelTripletMarginLoss {
public:
    __aicore__ inline KernelTripletMarginLoss() {}
    __aicore__ inline void Init(GM_ADDR anchor, GM_ADDR positive, GM_ADDR negative, GM_ADDR loss,
                                uint32_t batchSize, uint32_t dimSize, float margin)
    {
        this->batchSize = batchSize;
        this->dimSize = dimSize;
        this->margin = margin;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // Distribute batch samples across blocks
        this->batchPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startBatch = blockIdx * this->batchPerBlock;
        this->endBatch = startBatch + this->batchPerBlock;
        if (this->endBatch > batchSize) this->endBatch = batchSize;
        this->actualBatch = (this->endBatch > this->startBatch) ? (this->endBatch - this->startBatch) : 0;

        // Set up global memory pointers
        anchorGm.SetGlobalBuffer((__gm__ float*)anchor, batchSize * dimSize);
        positiveGm.SetGlobalBuffer((__gm__ float*)positive, batchSize * dimSize);
        negativeGm.SetGlobalBuffer((__gm__ float*)negative, batchSize * dimSize);
        lossGm.SetGlobalBuffer((__gm__ float*)loss, 1);

        // We process each sample's dimension in tiles
        // Align tile size to 32 bytes = 8 floats
        uint32_t alignNum = 8;

        // Determine tile size for processing dimension
        // We want to fit tiles in UB. Each tile needs space for anchor, positive, negative, diff, squared
        // Let's use a tile size that fits well
        uint32_t maxTileSize = 8192; // floats per tile
        if (dimSize <= maxTileSize) {
            this->tileDim = ((dimSize + alignNum - 1) / alignNum) * alignNum;
            this->tileNumDim = 1;
        } else {
            this->tileDim = maxTileSize;
            this->tileNumDim = (dimSize + maxTileSize - 1) / maxTileSize;
        }
        this->lastTileDim = dimSize - (this->tileNumDim - 1) * (this->tileNumDim > 1 ? maxTileSize : dimSize);
        if (this->tileNumDim == 1) {
            this->lastTileDim = dimSize;
        } else {
            this->lastTileDim = dimSize - (this->tileNumDim - 1) * maxTileSize;
        }

        // Allocate buffers: we need space for anchor_tile, pos_tile, neg_tile, diff, temp
        uint32_t bufBytes = this->tileDim * sizeof(float);
        pipe.InitBuffer(inQueueA, 1, bufBytes);
        pipe.InitBuffer(inQueueP, 1, bufBytes);
        pipe.InitBuffer(inQueueN, 1, bufBytes);
        pipe.InitBuffer(outQueue, 1, bufBytes);
        pipe.InitBuffer(tmpBuf1, 1, bufBytes);
        // For partial sums per sample and reduction
        uint32_t reduceBufSize = ((this->actualBatch + alignNum - 1) / alignNum) * alignNum;
        if (reduceBufSize < alignNum) reduceBufSize = alignNum;
        this->reduceBufLen = reduceBufSize;
        pipe.InitBuffer(tmpBuf2, 1, reduceBufSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->actualBatch == 0) return;

        uint32_t alignNum = 8;
        // We'll store per-sample loss in tmpBuf2
        AscendC::LocalTensor<float> sampleLoss = tmpBuf2.AllocTensor<float>();
        // zero it out
        AscendC::Duplicate(sampleLoss, 0.0f, this->reduceBufLen);

        for (uint32_t b = 0; b < this->actualBatch; b++) {
            uint32_t globalBatchIdx = this->startBatch + b;
            float distPos = 0.0f;
            float distNeg = 0.0f;

            // Process dimension in tiles
            uint32_t dimProcessed = 0;
            for (uint32_t t = 0; t < this->tileNumDim; t++) {
                uint32_t curDim;
                if (t == this->tileNumDim - 1) {
                    curDim = this->lastTileDim;
                } else {
                    curDim = (this->tileNumDim == 1) ? this->dimSize : 8192;
                }
                uint32_t curDimAligned = ((curDim + alignNum - 1) / alignNum) * alignNum;

                // Copy anchor tile
                AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                AscendC::LocalTensor<float> pLocal = inQueueP.AllocTensor<float>();
                AscendC::LocalTensor<float> nLocal = inQueueN.AllocTensor<float>();
                AscendC::LocalTensor<float> diffLocal = outQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> tmpLocal = tmpBuf1.AllocTensor<float>();

                uint32_t gmOffset = globalBatchIdx * this->dimSize + dimProcessed;
                AscendC::DataCopy(aLocal, anchorGm[gmOffset], curDimAligned);
                AscendC::DataCopy(pLocal, positiveGm[gmOffset], curDimAligned);
                AscendC::DataCopy(nLocal, negativeGm[gmOffset], curDimAligned);

                // Compute (anchor - positive)^2 sum
                AscendC::Sub(diffLocal, aLocal, pLocal, curDimAligned);
                AscendC::Mul(tmpLocal, diffLocal, diffLocal, curDimAligned);
                // Zero out padded elements if needed
                if (curDim < curDimAligned) {
                    for (uint32_t i = curDim; i < curDimAligned; i++) {
                        tmpLocal.SetValue(i, 0.0f);
                    }
                }
                // Reduce sum
                float sumPos = 0.0f;
                AscendC::ReduceSum(diffLocal, tmpLocal, tmpLocal, curDimAligned);
                sumPos = diffLocal.GetValue(0);
                distPos += sumPos;

                // Compute (anchor - negative)^2 sum
                AscendC::Sub(diffLocal, aLocal, nLocal, curDimAligned);
                AscendC::Mul(tmpLocal, diffLocal, diffLocal, curDimAligned);
                if (curDim < curDimAligned) {
                    for (uint32_t i = curDim; i < curDimAligned; i++) {
                        tmpLocal.SetValue(i, 0.0f);
                    }
                }
                float sumNeg = 0.0f;
                AscendC::ReduceSum(diffLocal, tmpLocal, tmpLocal, curDimAligned);
                sumNeg = diffLocal.GetValue(0);
                distNeg += sumNeg;

                dimProcessed += curDim;

                inQueueA.FreeTensor(aLocal);
                inQueueP.FreeTensor(pLocal);
                inQueueN.FreeTensor(nLocal);
                outQueue.FreeTensor(diffLocal);
                tmpBuf1.FreeTensor(tmpLocal);
            }

            // distPos = sqrt(distPos), distNeg = sqrt(distNeg)
            // loss_sample = max(0, sqrt(distPos) - sqrt(distNeg) + margin)
            float sqrtPos = 0.0f;
            float sqrtNeg = 0.0f;
            if (distPos > 0.0f) {
                // Manual sqrt approximation using Newton's method
                sqrtPos = distPos;
                for (int iter = 0; iter < 20; iter++) {
                    sqrtPos = 0.5f * (sqrtPos + distPos / sqrtPos);
                }
            }
            if (distNeg > 0.0f) {
                sqrtNeg = distNeg;
                for (int iter = 0; iter < 20; iter++) {
                    sqrtNeg = 0.5f * (sqrtNeg + distNeg / sqrtNeg);
                }
            }

            float lossVal = sqrtPos - sqrtNeg + this->margin;
            if (lossVal < 0.0f) lossVal = 0.0f;
            sampleLoss.SetValue(b, lossVal);
        }

        // Sum all sample losses and divide by batchSize for mean
        // We need the global mean, so each block computes its partial sum
        // and writes it, then we handle final reduction
        uint32_t alignNum = 8;
        AscendC::LocalTensor<float> tmpReduce = tmpBuf1.AllocTensor<float>();
        uint32_t reduceLen = this->reduceBufLen;
        // zero out padding
        for (uint32_t i = this->actualBatch; i < reduceLen; i++) {
            sampleLoss.SetValue(i, 0.0f);
        }
        if (reduceLen >= alignNum) {
            AscendC::ReduceSum(tmpReduce, sampleLoss, tmpReduce, reduceLen);
        } else {
            AscendC::ReduceSum(tmpReduce, sampleLoss, tmpReduce, alignNum);
        }
        float partialSum = tmpReduce.GetValue(0);
        float meanLoss = partialSum / (float)this->batchSize;

        // Use atomic add to accumulate to output
        // Only block 0 initializes the output
        tmpReduce.SetValue(0, meanLoss);
        // Pad for alignment
        for (uint32_t i = 1; i < alignNum; i++) {
            tmpReduce.SetValue(i, 0.0f);
        }

        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(lossGm[0], tmpReduce, alignNum);
        AscendC::SetAtomicNone();

        tmpBuf1.FreeTensor(tmpReduce);
        tmpBuf2.FreeTensor(sampleLoss);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> inQueueA, inQueueP, inQueueN;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> anchorGm, positiveGm, negativeGm, lossGm;
    uint32_t batchSize;
    uint32_t dimSize;
    float margin;
    uint32_t batchPerBlock;
    uint32_t startBatch;
    uint32_t endBatch;
    uint32_t actualBatch;
    uint32_t tileDim;
    uint32_t tileNumDim;
    uint32_t lastTileDim;
    uint32_t reduceBufLen;
};

extern "C" __global__ __aicore__ void triplet_margin_loss_custom(GM_ADDR anchor, GM_ADDR positive, GM_ADDR negative, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelTripletMarginLoss op;
    op.Init(anchor, positive, negative, loss, tiling_data.batchSize, tiling_data.dimSize, tiling_data.margin);
    op.Process();
}
