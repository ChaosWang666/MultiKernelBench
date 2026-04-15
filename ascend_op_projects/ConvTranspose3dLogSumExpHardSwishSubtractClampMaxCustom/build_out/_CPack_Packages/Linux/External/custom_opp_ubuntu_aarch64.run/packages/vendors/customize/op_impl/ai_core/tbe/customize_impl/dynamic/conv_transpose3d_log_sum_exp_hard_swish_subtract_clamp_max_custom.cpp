
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFused {
public:
    __aicore__ inline KernelFused() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t spatialSize, uint32_t totalOutput,
                                 uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->spatialSize = spatialSize;
        this->totalOutput = totalOutput;

        // Each block processes a portion of the output spatial elements
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->blockLength = totalOutput / blockNum;
        uint32_t remainder = totalOutput % blockNum;
        if (blockIdx < remainder) {
            this->blockLength += 1;
            this->blockOffset = blockIdx * this->blockLength;
        } else {
            this->blockOffset = blockIdx * this->blockLength + remainder;
        }

        if (this->blockLength == 0) {
            this->tileNum = 0;
            return;
        }

        // Align tileLength to 32 bytes (8 floats)
        this->tileNum = tileNum;
        if (this->tileNum > this->blockLength) {
            this->tileNum = this->blockLength;
        }
        this->tileLength = this->blockLength / this->tileNum;
        this->tailLength = this->blockLength - this->tileLength * this->tileNum;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * channels * spatialSize);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, 1);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockOffset, this->blockLength + this->tailLength);

        // We need buffers for one channel tile and accumulation
        uint32_t maxTile = this->tileLength;
        if (this->tailLength > maxTile) maxTile = this->tailLength;
        if (maxTile == 0) maxTile = 1;
        // Align to 8 floats (32 bytes)
        uint32_t alignedTile = (maxTile + 7) / 8 * 8;

        pipe.InitBuffer(inQueueX, 1, alignedTile * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, alignedTile * sizeof(float));
        // temp buffers for computation
        pipe.InitBuffer(tmpBuf1, 1, alignedTile * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, alignedTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;

        // Read bias value - we'll use it in computation
        // Process each tile
        for (uint32_t t = 0; t < this->tileNum; t++) {
            uint32_t curLen = this->tileLength;
            uint32_t outOffset = t * this->tileLength;
            ProcessTile(outOffset, curLen);
        }
        if (this->tailLength > 0) {
            uint32_t outOffset = this->tileNum * this->tileLength;
            ProcessTile(outOffset, this->tailLength);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t outOffset, uint32_t curLen)
    {
        uint32_t alignedLen = (curLen + 7) / 8 * 8;

        // Allocate accumulator for logsumexp
        AscendC::LocalTensor<float> accLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> chanLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1Local = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2Local = tmpBuf2.AllocTensor<float>();

        // First pass: find max across channels for numerical stability
        // Load channel 0
        for (uint32_t i = 0; i < curLen; i++) {
            uint32_t globalOutIdx = this->blockOffset + outOffset + i;
            uint32_t b = globalOutIdx / this->spatialSize;
            uint32_t s = globalOutIdx % this->spatialSize;
            uint32_t srcIdx = b * this->channels * this->spatialSize + s;
            chanLocal.SetValue(i, xGm.GetValue(srcIdx));
        }
        // Initialize acc with channel 0 values (as max)
        AscendC::DataCopy(accLocal, chanLocal, alignedLen);

        // Find max across all channels
        for (uint32_t c = 1; c < this->channels; c++) {
            for (uint32_t i = 0; i < curLen; i++) {
                uint32_t globalOutIdx = this->blockOffset + outOffset + i;
                uint32_t b = globalOutIdx / this->spatialSize;
                uint32_t s = globalOutIdx % this->spatialSize;
                uint32_t srcIdx = b * this->channels * this->spatialSize + c * this->spatialSize + s;
                chanLocal.SetValue(i, xGm.GetValue(srcIdx));
            }
            AscendC::Max(accLocal, accLocal, chanLocal, alignedLen);
        }

        // accLocal now has max values. Store in tmp2
        AscendC::DataCopy(tmp2Local, accLocal, alignedLen);

        // Second pass: compute sum of exp(x - max)
        // Channel 0
        for (uint32_t i = 0; i < curLen; i++) {
            uint32_t globalOutIdx = this->blockOffset + outOffset + i;
            uint32_t b = globalOutIdx / this->spatialSize;
            uint32_t s = globalOutIdx % this->spatialSize;
            uint32_t srcIdx = b * this->channels * this->spatialSize + s;
            chanLocal.SetValue(i, xGm.GetValue(srcIdx));
        }
        AscendC::Sub(chanLocal, chanLocal, tmp2Local, alignedLen);
        AscendC::Exp(chanLocal, chanLocal, alignedLen);
        AscendC::DataCopy(accLocal, chanLocal, alignedLen);

        for (uint32_t c = 1; c < this->channels; c++) {
            for (uint32_t i = 0; i < curLen; i++) {
                uint32_t globalOutIdx = this->blockOffset + outOffset + i;
                uint32_t b = globalOutIdx / this->spatialSize;
                uint32_t s = globalOutIdx % this->spatialSize;
                uint32_t srcIdx = b * this->channels * this->spatialSize + c * this->spatialSize + s;
                chanLocal.SetValue(i, xGm.GetValue(srcIdx));
            }
            AscendC::Sub(chanLocal, chanLocal, tmp2Local, alignedLen);
            AscendC::Exp(chanLocal, chanLocal, alignedLen);
            AscendC::Add(accLocal, accLocal, chanLocal, alignedLen);
        }

        // accLocal = sum of exp(x_c - max)
        // logsumexp = max + log(sum_exp)
        AscendC::Ln(accLocal, accLocal, alignedLen);
        AscendC::Add(accLocal, accLocal, tmp2Local, alignedLen);
        // accLocal now has logsumexp result

        // HardSwish: x * sigmoid(x + 3) / 6
        // = x * (1 / (1 + exp(-(x+3)))) / 6
        // Compute x + 3
        float three = 3.0f;
        AscendC::Adds(tmp1Local, accLocal, three, alignedLen);
        // Compute -(x+3)
        float negOne = -1.0f;
        AscendC::Muls(tmp1Local, tmp1Local, negOne, alignedLen);
        // exp(-(x+3))
        AscendC::Exp(tmp1Local, tmp1Local, alignedLen);
        // 1 + exp(-(x+3))
        float one = 1.0f;
        AscendC::Adds(tmp1Local, tmp1Local, one, alignedLen);
        // 1 / (1 + exp(-(x+3)))
        AscendC::Reciprocal(tmp1Local, tmp1Local, alignedLen);
        // x * sigmoid(x+3)
        AscendC::Mul(accLocal, accLocal, tmp1Local, alignedLen);
        // / 6
        float oneSixth = 1.0f / 6.0f;
        AscendC::Muls(accLocal, accLocal, oneSixth, alignedLen);

        // Subtract bias
        float biasVal = biasGm.GetValue(0);
        float negBias = -biasVal;
        AscendC::Adds(accLocal, accLocal, negBias, alignedLen);

        // Clamp to [-1, 1]
        float minVal = -1.0f;
        float maxVal = 1.0f;
        // clamp min: max(x, -1)
        AscendC::Maxs(accLocal, accLocal, minVal, alignedLen);
        // clamp max: min(x, 1)
        AscendC::Mins(accLocal, accLocal, maxVal, alignedLen);

        // Copy out
        for (uint32_t i = 0; i < curLen; i++) {
            zGm.SetValue(outOffset + i, accLocal.GetValue(i));
        }

        tmpBuf2.FreeTensor(tmp2Local);
        tmpBuf1.FreeTensor(tmp1Local);
        inQueueX.FreeTensor(chanLocal);
        outQueueZ.FreeTensor(accLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpBuf1;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t spatialSize;
    uint32_t totalOutput;
    uint32_t blockLength;
    uint32_t blockOffset;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_log_sum_exp_hard_swish_subtract_clamp_max_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFused op;
    op.Init(x, bias, z, tiling_data.batchSize, tiling_data.channels,
            tiling_data.spatialSize, tiling_data.totalOutput, tiling_data.tileNum);
    op.Process();
}
