
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelBatchNorm {
public:
    __aicore__ inline KernelBatchNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                 GM_ADDR runningMean, GM_ADDR runningVar,
                                 GM_ADDR y,
                                 uint32_t batchSize, uint32_t numFeatures,
                                 uint32_t height, uint32_t width, float eps)
    {
        this->batchSize = batchSize;
        this->numFeatures = numFeatures;
        this->height = height;
        this->width = width;
        this->eps = eps;
        this->spatialSize = height * width;
        this->totalElements = batchSize * numFeatures * spatialSize;

        // Each block processes a subset of (batch, channel) pairs
        uint32_t totalBC = batchSize * numFeatures;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->bcStart = blockIdx * ((totalBC + numBlocks - 1) / numBlocks);
        this->bcEnd = (blockIdx + 1) * ((totalBC + numBlocks - 1) / numBlocks);
        if (this->bcEnd > totalBC) this->bcEnd = totalBC;

        xGm.SetGlobalBuffer((__gm__ float *)x, this->totalElements);
        yGm.SetGlobalBuffer((__gm__ float *)y, this->totalElements);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, numFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, numFeatures);
        meanGm.SetGlobalBuffer((__gm__ float *)runningMean, numFeatures);
        varGm.SetGlobalBuffer((__gm__ float *)runningVar, numFeatures);

        // Compute tile parameters for spatial dimension processing
        // We process spatialSize elements per (batch, channel) pair
        // Align tileLength to 32 bytes / 4 bytes = 8 elements
        uint32_t alignNum = 8;
        this->tileNum = BUFFER_NUM;
        this->tileLength = (this->spatialSize + this->tileNum - 1) / this->tileNum;
        // align up tileLength
        this->tileLength = ((this->tileLength + alignNum - 1) / alignNum) * alignNum;
        // Recalculate: last tile may be smaller
        this->lastTileLength = this->spatialSize - this->tileLength * (this->tileNum - 1);
        if (this->lastTileLength <= 0) {
            this->tileNum = (this->spatialSize + this->tileLength - 1) / this->tileLength;
            this->lastTileLength = this->spatialSize - this->tileLength * (this->tileNum - 1);
        }
        this->alignedLastTile = ((this->lastTileLength + alignNum - 1) / alignNum) * alignNum;

        uint32_t bufSize = this->tileLength * sizeof(float);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, bufSize);
        pipe.InitBuffer(tmpBuf, 1, bufSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t bc = this->bcStart; bc < this->bcEnd; bc++) {
            uint32_t c = bc % this->numFeatures;
            uint32_t gmOffset = bc * this->spatialSize;

            // Load BN parameters for this channel
            float mean_val = 0.0f;
            float var_val = 0.0f;
            float w_val = 1.0f;
            float b_val = 0.0f;

            // We need to read single values from GM
            // Use a temporary local tensor approach
            {
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                // Read mean
                AscendC::DataCopy(tmpLocal, meanGm[c / 8 * 8], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                mean_val = tmpLocal.GetValue(c % 8);
                // Read var
                AscendC::DataCopy(tmpLocal, varGm[c / 8 * 8], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                var_val = tmpLocal.GetValue(c % 8);
                // Read weight
                AscendC::DataCopy(tmpLocal, weightGm[c / 8 * 8], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                w_val = tmpLocal.GetValue(c % 8);
                // Read bias
                AscendC::DataCopy(tmpLocal, biasGm[c / 8 * 8], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                b_val = tmpLocal.GetValue(c % 8);
                tmpBuf.FreeTensor(tmpLocal);
            }

            // Compute scale = weight / sqrt(var + eps) and shift = bias - mean * scale
            float invStd = 1.0f / sqrtf(var_val + this->eps);
            float scale = w_val * invStd;
            float shift = b_val - mean_val * scale;

            for (uint32_t t = 0; t < this->tileNum; t++) {
                uint32_t curTileLen = (t < this->tileNum - 1) ? this->tileLength : this->lastTileLength;
                uint32_t alignedLen = (t < this->tileNum - 1) ? this->tileLength : this->alignedLastTile;

                // CopyIn
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[gmOffset + t * this->tileLength], alignedLen);
                inQueueX.EnQue(xLocal);

                // Compute: y = x * scale + shift
                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
                AscendC::Muls(yLocal, xIn, scale, alignedLen);
                AscendC::Adds(yLocal, yLocal, shift, alignedLen);
                outQueueY.EnQue(yLocal);
                inQueueX.FreeTensor(xIn);

                // CopyOut
                AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
                AscendC::DataCopy(yGm[gmOffset + t * this->tileLength], yOut, alignedLen);
                outQueueY.FreeTensor(yOut);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm, yGm, weightGm, biasGm, meanGm, varGm;
    uint32_t batchSize, numFeatures, height, width;
    uint32_t spatialSize, totalElements;
    uint32_t bcStart, bcEnd;
    uint32_t tileNum, tileLength, lastTileLength, alignedLastTile;
    float eps;
};

extern "C" __global__ __aicore__ void batch_norm_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                                         GM_ADDR running_mean, GM_ADDR running_var,
                                                         GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBatchNorm op;
    op.Init(x, weight, bias, running_mean, running_var, y,
            tiling_data.batchSize, tiling_data.numFeatures,
            tiling_data.height, tiling_data.width, tiling_data.eps);
    op.Process();
}
