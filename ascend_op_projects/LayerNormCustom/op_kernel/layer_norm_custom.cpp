
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelLayerNorm {
public:
    __aicore__ inline KernelLayerNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                uint32_t batchSize, uint32_t normSize, uint32_t tileNum, float epsilon)
    {
        this->batchSize = batchSize;
        this->normSize = normSize;
        this->tileNum = tileNum;
        this->epsilon = epsilon;

        // Each block processes a subset of batches
        uint32_t totalBatches = batchSize;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->batchPerBlock = (totalBatches + numBlocks - 1) / numBlocks;
        this->batchStart = blockIdx * this->batchPerBlock;
        if (this->batchStart + this->batchPerBlock > totalBatches) {
            this->batchPerBlock = (this->batchStart < totalBatches) ? (totalBatches - this->batchStart) : 0;
        }

        // Align normSize to 32 bytes (8 floats)
        this->alignedNormSize = (normSize + 7) / 8 * 8;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * normSize);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, normSize);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, normSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * normSize);

        // We process normSize elements per batch instance
        // Use tileNum to split normSize if it's large
        this->tileLengthRaw = normSize;
        this->tileLengthAligned = this->alignedNormSize;

        pipe.InitBuffer(inQueueX, 1, this->tileLengthAligned * sizeof(float));
        pipe.InitBuffer(inQueueGamma, 1, this->tileLengthAligned * sizeof(float));
        pipe.InitBuffer(inQueueBeta, 1, this->tileLengthAligned * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, this->tileLengthAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf1, 1, this->tileLengthAligned * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, this->tileLengthAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->batchPerBlock == 0) return;
        for (uint32_t i = 0; i < this->batchPerBlock; i++) {
            uint32_t batchIdx = this->batchStart + i;
            if (batchIdx >= this->batchSize) break;
            CopyIn(batchIdx);
            Compute(batchIdx);
            CopyOut(batchIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();

        // Zero out the buffer first if normSize is not aligned
        if (this->tileLengthAligned > this->tileLengthRaw) {
            AscendC::Duplicate(xLocal, (float)0, this->tileLengthAligned);
            AscendC::Duplicate(gammaLocal, (float)0, this->tileLengthAligned);
            AscendC::Duplicate(betaLocal, (float)0, this->tileLengthAligned);
        }

        AscendC::DataCopy(xLocal, xGm[batchIdx * this->normSize], this->tileLengthAligned);
        AscendC::DataCopy(gammaLocal, gammaGm[0], this->tileLengthAligned);
        AscendC::DataCopy(betaLocal, betaGm[0], this->tileLengthAligned);

        inQueueX.EnQue(xLocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void Compute(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();

        uint32_t len = this->tileLengthAligned;
        float invN = 1.0f / (float)this->normSize;

        // Compute mean: sum(x) / N
        // Use ReduceSum
        float meanVal = 0.0f;
        AscendC::ReduceSum(tmp1, xLocal, tmp2, len);
        meanVal = tmp1.GetValue(0) * invN;

        // Compute (x - mean)
        AscendC::Adds(yLocal, xLocal, -meanVal, len);

        // Compute variance: sum((x - mean)^2) / N
        AscendC::Mul(tmp1, yLocal, yLocal, len);
        AscendC::ReduceSum(tmp2, tmp1, tmp1, len);
        float varVal = tmp2.GetValue(0) * invN;

        // Compute 1/sqrt(var + eps)
        float invStd = 1.0f / sqrtf(varVal + this->epsilon);

        // Normalize: (x - mean) * invStd
        AscendC::Muls(yLocal, yLocal, invStd, len);

        // Apply gamma and beta: y = gamma * normalized + beta
        AscendC::Mul(yLocal, yLocal, gammaLocal, len);
        AscendC::Add(yLocal, yLocal, betaLocal, len);

        outQueueY.EnQue(yLocal);
        tmpBuf1.FreeTensor(tmp1);
        tmpBuf2.FreeTensor(tmp2);
        inQueueX.FreeTensor(xLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[batchIdx * this->normSize], yLocal, this->tileLengthAligned);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueGamma, inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm, gammaGm, betaGm, yGm;
    uint32_t batchSize;
    uint32_t normSize;
    uint32_t tileNum;
    float epsilon;
    uint32_t batchPerBlock;
    uint32_t batchStart;
    uint32_t alignedNormSize;
    uint32_t tileLengthRaw;
    uint32_t tileLengthAligned;
};

extern "C" __global__ __aicore__ void layer_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLayerNorm op;
    op.Init(x, gamma, beta, y, tiling_data.batchSize, tiling_data.normSize, tiling_data.tileNum, tiling_data.epsilon);
    op.Process();
}
