
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelScaleBnAvgPool {
public:
    __aicore__ inline KernelScaleBnAvgPool() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                                GM_ADDR running_mean, GM_ADDR running_var,
                                GM_ADDR z,
                                uint32_t batchSize, uint32_t channels,
                                uint32_t spatialSize, float scaleFactor,
                                float eps, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->spatialSize = spatialSize;
        this->scaleFactor = scaleFactor;
        this->eps = eps;
        this->tileNum = tileNum;

        // Total work items = batchSize * channels
        // Each block handles a subset of (batch, channel) pairs
        uint32_t totalBC = batchSize * channels;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->bcStart = (totalBC / blockNum) * blockIdx + (blockIdx < (totalBC % blockNum) ? blockIdx : (totalBC % blockNum));
        uint32_t bcCount = totalBC / blockNum + (blockIdx < (totalBC % blockNum) ? 1 : 0);
        this->bcEnd = this->bcStart + bcCount;

        // Global memory setup
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * channels * spatialSize);
        weightGm.SetGlobalBuffer((__gm__ float *)weight, channels);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, channels);
        meanGm.SetGlobalBuffer((__gm__ float *)running_mean, channels);
        varGm.SetGlobalBuffer((__gm__ float *)running_var, channels);
        zGm.SetGlobalBuffer((__gm__ float *)z, batchSize * channels);

        // Determine tile length for spatial dimension processing
        // We process spatialSize elements per (batch, channel) pair
        uint32_t alignedSpatial = ((spatialSize + 7) / 8) * 8;
        if (alignedSpatial < 8) alignedSpatial = 8;

        // Determine number of tiles for spatial dimension
        this->spatialTiles = tileNum;
        if (this->spatialTiles > spatialSize) this->spatialTiles = 1;

        this->tileLength = spatialSize / this->spatialTiles;
        this->lastTileLength = spatialSize - this->tileLength * (this->spatialTiles - 1);

        // Align tile lengths to 8 elements (32 bytes for float)
        uint32_t alignedTileLen = ((this->tileLength + 7) / 8) * 8;
        uint32_t alignedLastTileLen = ((this->lastTileLength + 7) / 8) * 8;
        uint32_t maxAlignedLen = alignedTileLen > alignedLastTileLen ? alignedTileLen : alignedLastTileLen;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, maxAlignedLen * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, 8 * sizeof(float)); // just for one output value
        // Workspace for intermediate results
        pipe.InitBuffer(tmpBuf, 1, maxAlignedLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t bc = this->bcStart; bc < this->bcEnd; bc++) {
            uint32_t b = bc / this->channels;
            uint32_t c = bc % this->channels;
            ProcessOneBC(b, c);
        }
    }

private:
    __aicore__ inline void ProcessOneBC(uint32_t b, uint32_t c)
    {
        // Load BN parameters for this channel
        float gamma = weightGm.GetValue(c);
        float beta = biasGm.GetValue(c);
        float mean = meanGm.GetValue(c);
        float var = varGm.GetValue(c);

        // Precompute: combined_scale = scale_factor * gamma / sqrt(var + eps)
        // combined_bias = beta - mean * scale_factor * gamma / sqrt(var + eps)
        float invStd = 1.0f / sqrt(var + this->eps);
        float combinedScale = this->scaleFactor * gamma * invStd;
        float combinedBias = beta - mean * this->scaleFactor * gamma * invStd;

        uint32_t offset = (b * this->channels + c) * this->spatialSize;
        float sum = 0.0f;

        for (uint32_t t = 0; t < this->spatialTiles; t++) {
            uint32_t curTileLen = (t == this->spatialTiles - 1) ? this->lastTileLength : this->tileLength;
            uint32_t alignedLen = ((curTileLen + 7) / 8) * 8;
            uint32_t tileOffset = offset + t * this->tileLength;

            // CopyIn
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            // Zero out padding
            if (alignedLen > curTileLen) {
                for (uint32_t i = curTileLen; i < alignedLen; i++) {
                    xLocal.SetValue(i, 0.0f);
                }
            }
            AscendC::DataCopy(xLocal, xGm[tileOffset], alignedLen);
            inQueueX.EnQue(xLocal);

            // Compute: apply scale + BN: y = x * combinedScale + combinedBias
            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

            AscendC::Muls(tmpLocal, xIn, combinedScale, alignedLen);
            AscendC::Adds(tmpLocal, tmpLocal, combinedBias, alignedLen);

            // Sum for average pooling
            // Zero out padding values before summing
            if (alignedLen > curTileLen) {
                for (uint32_t i = curTileLen; i < alignedLen; i++) {
                    tmpLocal.SetValue(i, 0.0f);
                }
            }

            // Reduce sum
            float tileSum = 0.0f;
            // Use vector reduction if possible
            AscendC::LocalTensor<float> workLocal = xIn; // reuse buffer
            AscendC::ReduceSum(workLocal, tmpLocal, tmpLocal, alignedLen);
            tileSum = workLocal.GetValue(0);

            sum += tileSum;

            inQueueX.FreeTensor(xIn);
        }

        // Global average pool: divide by spatialSize
        float avgVal = sum / (float)this->spatialSize;

        // Write output
        uint32_t outIdx = b * this->channels + c;
        // Need to write a single value but DataCopy requires aligned access
        // Use scalar write
        zGm.SetValue(outIdx, avgVal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> weightGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> meanGm;
    AscendC::GlobalTensor<float> varGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t spatialSize;
    float scaleFactor;
    float eps;
    uint32_t tileNum;
    uint32_t spatialTiles;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t bcStart;
    uint32_t bcEnd;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scale_batch_norm_global_avg_pool_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR running_mean, GM_ADDR running_var,
    GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelScaleBnAvgPool op;
    op.Init(x, weight, bias, running_mean, running_var, z,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.spatialSize, tiling_data.scaleFactor,
            tiling_data.eps, tiling_data.tileNum);
    op.Process();
}
