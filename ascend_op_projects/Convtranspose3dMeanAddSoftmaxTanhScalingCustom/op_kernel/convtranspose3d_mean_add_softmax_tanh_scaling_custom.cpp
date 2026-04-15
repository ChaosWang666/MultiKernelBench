
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// Each block handles a subset of (batch, spatial) positions.
// For each position (b, h, w), we compute:
// 1. mean over D for each channel
// 2. add bias
// 3. softmax over channels
// 4. tanh
// 5. scale

class KernelMeanAddSoftmaxTanhScale {
public:
    __aicore__ inline KernelMeanAddSoftmaxTanhScale() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t channels, uint32_t depth,
                                 uint32_t height, uint32_t width, float scalingFactor, uint32_t tileNum)
    {
        this->B = batchSize;
        this->C = channels;
        this->D = depth;
        this->H = height;
        this->W = width;
        this->scalingFactor = scalingFactor;
        this->HW = H * W;
        this->DHW = D * H * W;

        // Total spatial positions across all batches: B * H * W
        uint32_t totalPositions = B * HW;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->posStart = (totalPositions / blockNum) * blockIdx;
        uint32_t posEnd = (blockIdx == blockNum - 1) ? totalPositions : (totalPositions / blockNum) * (blockIdx + 1);
        this->posCount = posEnd - this->posStart;

        // Align C to 8 for float (32 bytes)
        this->alignedC = ((C + 7) / 8) * 8;

        xGm.SetGlobalBuffer((__gm__ float *)x, B * C * DHW);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, C);
        zGm.SetGlobalBuffer((__gm__ float *)z, B * C * HW);

        // We need buffers for: channel vector read from x (for accumulation), bias, intermediate computations
        // inQueueX: for reading depth slices per channel group
        // outQueueZ: for writing output
        pipe.InitBuffer(inQueueX, 1, alignedC * sizeof(float));
        pipe.InitBuffer(inQueueY, 1, alignedC * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, alignedC * sizeof(float));
        pipe.InitBuffer(tmpBuf1, 1, alignedC * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, alignedC * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < posCount; i++) {
            uint32_t globalPos = posStart + i;
            uint32_t b = globalPos / HW;
            uint32_t hw = globalPos % HW;
            uint32_t h = hw / W;
            uint32_t w = hw % W;

            ComputePosition(b, h, w);
        }
    }

private:
    __aicore__ inline void ComputePosition(uint32_t b, uint32_t h, uint32_t w)
    {
        // Step 1: Compute mean over depth for each channel
        // accumulator
        AscendC::LocalTensor<float> accLocal = inQueueX.AllocTensor<float>();

        // Zero the accumulator
        AscendC::Duplicate<float>(accLocal, 0.0f, alignedC);

        AscendC::LocalTensor<float> tmpLocal = tmpBuf1.AllocTensor<float>();

        for (uint32_t d = 0; d < D; d++) {
            // For each channel c, x[b, c, d, h, w] is at offset: b*C*DHW + c*DHW + d*HW + h*W + w
            // We need to gather C values with stride DHW
            // Since channels are not contiguous in memory for a given spatial position, we load one by one
            for (uint32_t c = 0; c < C; c++) {
                uint32_t idx = b * C * DHW + c * DHW + d * HW + h * W + w;
                // We'll use scalar approach - copy single values
                AscendC::DataCopy(tmpLocal[c], xGm[idx], 1);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Add(accLocal, accLocal, tmpLocal, alignedC);
        }

        // Divide by D to get mean
        float invD = 1.0f / (float)D;
        AscendC::Muls(accLocal, accLocal, invD, alignedC);

        // Step 2: Add bias - bias shape (1, C, 1, 1, 1), stored as C floats
        AscendC::LocalTensor<float> biasLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(biasLocal, biasGm[0], alignedC);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Add(accLocal, accLocal, biasLocal, alignedC);

        // Step 3: Softmax over channels
        // Find max for numerical stability
        AscendC::LocalTensor<float> tmp2Local = tmpBuf2.AllocTensor<float>();

        // ReduceMax
        float maxVal;
        AscendC::ReduceMax(tmpLocal, accLocal, alignedC, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        maxVal = tmpLocal.GetValue(0);

        // Subtract max
        AscendC::Adds(accLocal, accLocal, -maxVal, alignedC);

        // Exp
        AscendC::Exp(accLocal, accLocal, alignedC);

        // Sum
        float sumVal;
        AscendC::ReduceSum(tmpLocal, accLocal, alignedC, false);
        AscendC::PipeBarrier<PIPE_ALL>();
        sumVal = tmpLocal.GetValue(0);

        // Divide by sum
        float invSum = 1.0f / sumVal;
        AscendC::Muls(accLocal, accLocal, invSum, alignedC);

        // Step 4: Tanh
        // tanh(x) for softmax outputs (0,1) => tanh(small positive) ≈ small positive
        // Use the Tanh intrinsic if available, or compute manually
        // For AscendC, we can approximate or use available math ops

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        AscendC::Muls(tmp2Local, accLocal, 2.0f, alignedC);
        AscendC::Exp(tmp2Local, tmp2Local, alignedC);  // exp(2x)
        AscendC::Adds(tmpLocal, tmp2Local, -1.0f, alignedC);  // exp(2x) - 1
        AscendC::Adds(tmp2Local, tmp2Local, 1.0f, alignedC);  // exp(2x) + 1
        AscendC::Div(accLocal, tmpLocal, tmp2Local, alignedC); // tanh

        // Step 5: Scale
        AscendC::Muls(accLocal, accLocal, scalingFactor, alignedC);

        // Write output: z[b, c, 0, h, w] at offset b*C*HW + c*HW + h*W + w
        AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
        AscendC::DataCopy(outLocal, accLocal, alignedC);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t c = 0; c < C; c++) {
            uint32_t outIdx = b * C * HW + c * HW + h * W + w;
            AscendC::DataCopy(zGm[outIdx], outLocal[c], 1);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        tmpBuf2.FreeTensor(tmp2Local);
        tmpBuf1.FreeTensor(tmpLocal);
        inQueueY.FreeTensor(biasLocal);
        inQueueX.FreeTensor(accLocal);
        outQueueZ.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;

    uint32_t B, C, D, H, W;
    uint32_t HW, DHW;
    uint32_t alignedC;
    float scalingFactor;
    uint32_t posStart;
    uint32_t posCount;
};

extern "C" __global__ __aicore__ void convtranspose3d_mean_add_softmax_tanh_scaling_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMeanAddSoftmaxTanhScale op;
    op.Init(x, bias, z, workspace,
            tiling_data.batchSize, tiling_data.channels, tiling_data.depth,
            tiling_data.height, tiling_data.width, tiling_data.scalingFactor,
            tiling_data.tileNum);
    op.Process();
}
