
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMinSoftmax {
public:
    __aicore__ inline KernelMinSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t channels,
                                 uint32_t dimD, uint32_t height, uint32_t width,
                                 uint32_t totalOutput, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->dimD = dimD;
        this->height = height;
        this->width = width;
        this->hw = height * width;
        this->totalOutput = totalOutput;
        this->totalInput = batchSize * channels * dimD * height * width;

        // Each block handles a portion of (batch * height * width) positions
        uint32_t totalPositions = batchSize * height * width;
        this->blockLength = totalPositions / AscendC::GetBlockNum();
        uint32_t remainder = totalPositions % AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < remainder) {
            this->blockLength += 1;
            this->startPos = blockIdx * this->blockLength;
        } else {
            this->startPos = blockIdx * this->blockLength + remainder;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, this->totalInput);
        yGm.SetGlobalBuffer((__gm__ float *)y, this->totalOutput);

        // We process one spatial position at a time across all channels
        // For each position, we need channels floats for min result, channels for softmax
        uint32_t alignedChannels = ((channels + 7) / 8) * 8;
        pipe.InitBuffer(inQueueX, 1, alignedChannels * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, alignedChannels * sizeof(float));
        pipe.InitBuffer(tmpBuf1, 1, alignedChannels * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, alignedChannels * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->blockLength; i++) {
            uint32_t pos = this->startPos + i;
            uint32_t b = pos / (this->hw);
            uint32_t rem = pos % (this->hw);
            uint32_t h = rem / this->width;
            uint32_t w = rem % this->width;
            ComputeMinSoftmax(b, h, w);
        }
    }

private:
    __aicore__ inline void ComputeMinSoftmax(uint32_t b, uint32_t h, uint32_t w)
    {
        uint32_t alignedChannels = ((channels + 7) / 8) * 8;

        // Step 1: Compute min along dimD for each channel
        AscendC::LocalTensor<float> minResult = inQueueX.AllocTensor<float>();

        // Initialize with first depth slice
        // Input layout: (batch, channels, D, H, W) contiguous
        // Index: b * (C*D*H*W) + c * (D*H*W) + d * (H*W) + h * W + w
        for (uint32_t c = 0; c < channels; c++) {
            uint32_t idx0 = b * (channels * dimD * hw) + c * (dimD * hw) + 0 * hw + h * width + w;
            float minVal = xGm.GetValue(idx0);
            for (uint32_t d = 1; d < dimD; d++) {
                uint32_t idx = b * (channels * dimD * hw) + c * (dimD * hw) + d * hw + h * width + w;
                float val = xGm.GetValue(idx);
                if (val < minVal) {
                    minVal = val;
                }
            }
            minResult.SetValue(c, minVal);
        }
        // Pad remaining
        for (uint32_t c = channels; c < alignedChannels; c++) {
            minResult.SetValue(c, (float)(-1e30));
        }
        inQueueX.EnQue(minResult);

        // Step 2: Softmax along channel dimension
        AscendC::LocalTensor<float> minLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();

        // Find max for numerical stability
        AscendC::ReduceMax(tmp1, minLocal, tmp2, alignedChannels);
        float maxVal = tmp1.GetValue(0);

        // If we have padding, recompute max over only valid channels
        if (alignedChannels > channels) {
            maxVal = minLocal.GetValue(0);
            for (uint32_t c = 1; c < channels; c++) {
                float v = minLocal.GetValue(c);
                if (v > maxVal) maxVal = v;
            }
        }

        // Subtract max
        AscendC::Adds(tmp1, minLocal, -maxVal, alignedChannels);

        // Exp
        AscendC::Exp(outLocal, tmp1, alignedChannels);

        // Zero out padding
        for (uint32_t c = channels; c < alignedChannels; c++) {
            outLocal.SetValue(c, 0.0f);
        }

        // Sum
        float sumExp = 0.0f;
        for (uint32_t c = 0; c < channels; c++) {
            sumExp += outLocal.GetValue(c);
        }

        // Divide
        float invSum = 1.0f / sumExp;
        AscendC::Muls(tmp1, outLocal, invSum, alignedChannels);

        // Write output: output layout (batch, channels, H, W)
        // Index: b * (C*H*W) + c * (H*W) + h * W + w
        for (uint32_t c = 0; c < channels; c++) {
            uint32_t outIdx = b * (channels * hw) + c * hw + h * width + w;
            yGm.SetValue(outIdx, tmp1.GetValue(c));
        }

        tmpBuf2.FreeTensor(tmp2);
        tmpBuf1.FreeTensor(tmp1);
        outQueueZ.FreeTensor(outLocal);
        inQueueX.FreeTensor(minLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t dimD;
    uint32_t height;
    uint32_t width;
    uint32_t hw;
    uint32_t totalOutput;
    uint32_t totalInput;
    uint32_t blockLength;
    uint32_t startPos;
};

extern "C" __global__ __aicore__ void conv3d_min_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinSoftmax op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.channels, tiling_data.dimD,
            tiling_data.height, tiling_data.width, tiling_data.totalOutput, tiling_data.tileNum);
    op.Process();
}
