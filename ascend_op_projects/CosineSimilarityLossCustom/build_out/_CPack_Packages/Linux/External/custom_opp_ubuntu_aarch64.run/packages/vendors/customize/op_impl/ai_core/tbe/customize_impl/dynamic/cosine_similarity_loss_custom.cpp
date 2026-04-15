
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelCosineSimilarityLoss {
public:
    __aicore__ inline KernelCosineSimilarityLoss() {}
    __aicore__ inline void Init(GM_ADDR predictions, GM_ADDR targets, GM_ADDR output,
                                 uint32_t batchSize, uint32_t featureSize)
    {
        this->batchSize = batchSize;
        this->featureSize = featureSize;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // Distribute batch rows among blocks
        this->batchPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startBatch = blockIdx * this->batchPerBlock;
        this->endBatch = startBatch + batchPerBlock;
        if (this->endBatch > batchSize) {
            this->endBatch = batchSize;
        }
        if (this->startBatch >= batchSize) {
            this->startBatch = batchSize;
            this->endBatch = batchSize;
        }

        // Align tile length to 32 bytes = 8 floats
        uint32_t alignNum = 8;
        this->alignedFeatureSize = ((featureSize + alignNum - 1) / alignNum) * alignNum;

        predGm.SetGlobalBuffer((__gm__ float*)predictions, batchSize * featureSize);
        targGm.SetGlobalBuffer((__gm__ float*)targets, batchSize * featureSize);
        outputGm.SetGlobalBuffer((__gm__ float*)output, 1);

        // We process one row at a time; need buffers for tiles of a row
        // For large featureSize, we tile within a row
        this->tileSize = alignedFeatureSize;
        // Limit tile size to fit in local memory (UB). We have ~248KB UB.
        // We need 4 buffers: pred tile, targ tile, temp1, temp2 -> 4 * tileSize * 4 bytes
        // Plus reduction workspace. Let's cap at 8192 floats per tile.
        uint32_t maxTileFloats = 8192;
        if (this->tileSize > maxTileFloats) {
            this->tileSize = (maxTileFloats / alignNum) * alignNum;
        }
        this->tileCount = (alignedFeatureSize + this->tileSize - 1) / this->tileSize;

        pipe.InitBuffer(inQueuePred, BUFFER_NUM, this->tileSize * sizeof(float));
        pipe.InitBuffer(inQueueTarg, BUFFER_NUM, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->tileSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        float localSum = 0.0f;

        for (uint32_t b = this->startBatch; b < this->endBatch; b++) {
            float dotProd = 0.0f;
            float normPred = 0.0f;
            float normTarg = 0.0f;

            for (uint32_t t = 0; t < this->tileCount; t++) {
                uint32_t offset = b * featureSize + t * this->tileSize;
                uint32_t remaining = featureSize - t * this->tileSize;
                uint32_t curTile = remaining < this->tileSize ? remaining : this->tileSize;
                // Align curTile for DataCopy
                uint32_t alignNum = 8;
                uint32_t copyLen = ((curTile + alignNum - 1) / alignNum) * alignNum;
                if (offset + copyLen > batchSize * featureSize + alignNum) {
                    copyLen = curTile;
                }

                // CopyIn predictions
                AscendC::LocalTensor<float> predLocal = inQueuePred.AllocTensor<float>();
                AscendC::DataCopy(predLocal, predGm[offset], copyLen);
                inQueuePred.EnQue(predLocal);

                // CopyIn targets
                AscendC::LocalTensor<float> targLocal = inQueueTarg.AllocTensor<float>();
                AscendC::DataCopy(targLocal, targGm[offset], copyLen);
                inQueueTarg.EnQue(targLocal);

                // Compute dot product contribution
                predLocal = inQueuePred.DeQue<float>();
                targLocal = inQueueTarg.DeQue<float>();
                AscendC::LocalTensor<float> tempLocal = outQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();

                // Zero out elements beyond curTile if curTile < copyLen
                if (curTile < copyLen) {
                    for (uint32_t i = curTile; i < copyLen; i++) {
                        predLocal.SetValue(i, 0.0f);
                        targLocal.SetValue(i, 0.0f);
                    }
                }

                // dot product: pred * targ
                AscendC::Mul(tempLocal, predLocal, targLocal, copyLen);
                float tDot = 0.0f;
                // Reduction
                AscendC::ReduceSum(tmpLocal, tempLocal, tmpLocal, copyLen);
                tDot = tmpLocal.GetValue(0);
                dotProd += tDot;

                // norm pred: pred * pred
                AscendC::Mul(tempLocal, predLocal, predLocal, copyLen);
                AscendC::ReduceSum(tmpLocal, tempLocal, tmpLocal, copyLen);
                normPred += tmpLocal.GetValue(0);

                // norm targ: targ * targ
                AscendC::Mul(tempLocal, targLocal, targLocal, copyLen);
                AscendC::ReduceSum(tmpLocal, tempLocal, tmpLocal, copyLen);
                normTarg += tmpLocal.GetValue(0);

                tmpBuf.FreeTensor(tmpLocal);
                outQueue.FreeTensor(tempLocal);
                inQueuePred.FreeTensor(predLocal);
                inQueueTarg.FreeTensor(targLocal);
            }

            // cosine_sim = dot / (norm_pred * norm_targ + eps)
            float eps = 1e-8f;
            float normP = normPred;
            float normT = normTarg;
            // sqrt
            float sqrtP = 0.0f;
            float sqrtT = 0.0f;
            // Manual sqrt approximation using iterative method
            if (normP > 0.0f) {
                sqrtP = normP;
                for (int i = 0; i < 20; i++) {
                    sqrtP = 0.5f * (sqrtP + normP / sqrtP);
                }
            }
            if (normT > 0.0f) {
                sqrtT = normT;
                for (int i = 0; i < 20; i++) {
                    sqrtT = 0.5f * (sqrtT + normT / sqrtT);
                }
            }

            float cosineSim = dotProd / (sqrtP * sqrtT + eps);
            localSum += (1.0f - cosineSim);
        }

        // Write partial sum - we need atomic add or use a simple approach
        // Since output is scalar, each block writes to a unique temp location
        // We'll use a simpler approach: block 0 location stores result
        // Actually, let's just use the global memory with block index offset and reduce on host
        // For simplicity, we accumulate to output using atomic-like pattern

        // Write local result; we'll average across all batches
        // Each block contributes its partial sum; we need to atomically add
        // Simple approach: write partial results, but since we only have 1 output element,
        // let's have each core do atomic add simulation through workspace or 
        // have block 0 do everything if batch is small enough

        // For correctness, use SetAtomicAdd
        float result = localSum / (float)batchSize;

        AscendC::LocalTensor<float> resultLocal = outQueue.AllocTensor<float>();
        resultLocal.SetValue(0, result);
        // Pad rest to zero
        for (uint32_t i = 1; i < 8; i++) {
            resultLocal.SetValue(i, 0.0f);
        }
        outQueue.EnQue(resultLocal);
        resultLocal = outQueue.DeQue<float>();

        AscendC::SetAtomicAdd<float>();
        AscendC::DataCopy(outputGm[0], resultLocal, 8);
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
    AscendC::GlobalTensor<float> outputGm;
    uint32_t batchSize;
    uint32_t featureSize;
    uint32_t batchPerBlock;
    uint32_t startBatch;
    uint32_t endBatch;
    uint32_t alignedFeatureSize;
    uint32_t tileSize;
    uint32_t tileCount;
};

extern "C" __global__ __aicore__ void cosine_similarity_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCosineSimilarityLoss op;
    op.Init(predictions, targets, output, tiling_data.batchSize, tiling_data.featureSize);
    op.Process();
}
