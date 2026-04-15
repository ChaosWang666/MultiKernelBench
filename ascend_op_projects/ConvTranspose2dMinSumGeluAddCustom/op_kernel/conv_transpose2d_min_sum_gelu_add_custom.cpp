
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelMinSumGeluAdd {
public:
    __aicore__ inline KernelMinSumGeluAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t height, uint32_t width, uint32_t biasLength)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->height = height;
        this->width = width;
        this->biasLength = biasLength;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        // Distribute batches across blocks
        this->batchPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->batchStart = blockIdx * this->batchPerBlock;
        this->batchEnd = this->batchStart + this->batchPerBlock;
        if (this->batchEnd > batchSize) this->batchEnd = batchSize;

        uint32_t batchElements = channels * height * width;
        uint32_t outElements = width; // per batch output is [1,1,width]

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * batchElements);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, biasLength);
        zGm.SetGlobalBuffer((__gm__ float*)z, batchSize * outElements);

        // We process one row (width elements) at a time
        // Align width to 32 bytes = 8 floats
        uint32_t alignedWidth = ((width + 7) / 8) * 8;
        this->alignedWidth = alignedWidth;

        // Buffer for one row of data
        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedWidth * sizeof(float));
        pipe.InitBuffer(minQueue, BUFFER_NUM, alignedWidth * sizeof(float));
        pipe.InitBuffer(sumQueue, BUFFER_NUM, alignedWidth * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedWidth * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t b = this->batchStart; b < this->batchEnd; b++) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx)
    {
        uint32_t batchOffset = batchIdx * channels * height * width;

        // Step 1: For each (h, w) position, compute min across channels
        // Then sum the min values across height dimension for each w
        // Result: [1, 1, 1, width]

        // Initialize sum accumulator
        LocalTensor<float> sumLocal = sumQueue.AllocTensor<float>();
        // Zero out sum buffer
        Duplicate(sumLocal, (float)0.0f, this->alignedWidth);

        for (uint32_t h = 0; h < height; h++) {
            // Initialize min buffer with first channel's row
            LocalTensor<float> minLocal = minQueue.AllocTensor<float>();
            uint32_t srcOffset = batchOffset + 0 * height * width + h * width;
            DataCopy(minLocal, xGm[srcOffset], this->alignedWidth);

            pipe_barrier(PIPE_ALL);

            // Iterate over remaining channels to find min
            for (uint32_t c = 1; c < channels; c++) {
                LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
                uint32_t cOffset = batchOffset + c * height * width + h * width;
                DataCopy(inLocal, xGm[cOffset], this->alignedWidth);
                pipe_barrier(PIPE_ALL);
                Min(minLocal, minLocal, inLocal, this->alignedWidth);
                pipe_barrier(PIPE_ALL);
                inQueue.FreeTensor(inLocal);
            }

            // Accumulate min values into sum
            Add(sumLocal, sumLocal, minLocal, this->alignedWidth);
            pipe_barrier(PIPE_ALL);
            minQueue.FreeTensor(minLocal);
        }

        // Step 2: Apply GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        // Approximate GELU using: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Or use the simpler sigmoid approximation: x * sigmoid(1.702 * x)
        // Let's use the tanh approximation for standard GELU
        
        LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // But we can use built-in Activation or manual computation
        // Let's compute step by step
        
        // tmp = x^2
        LocalTensor<float> tmpLocal = inQueue.AllocTensor<float>();
        Mul(tmpLocal, sumLocal, sumLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        // tmp = x^3
        Mul(tmpLocal, tmpLocal, sumLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        // tmp = 0.044715 * x^3
        Muls(tmpLocal, tmpLocal, (float)0.044715f, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        // tmp = x + 0.044715 * x^3
        Add(tmpLocal, sumLocal, tmpLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        // tmp = sqrt(2/pi) * (x + 0.044715 * x^3), sqrt(2/pi) ≈ 0.7978845608
        Muls(tmpLocal, tmpLocal, (float)0.7978845608f, this->alignedWidth);
        pipe_barrier(PIPE_ALL);

        // outLocal = tanh(tmp) - use series or built-in
        // AscendC has Tanh
        // First copy to a different tensor for tanh
        DataCopy(outLocal, tmpLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);

        // Use the Exp approach: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // Or simply use the built-in if available
        // AscendC should have Tanh API
        // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
        
        // Let's try to compute tanh manually:
        // exp_val = exp(2*x)
        Muls(outLocal, tmpLocal, (float)2.0f, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        Exp(outLocal, outLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        
        // numerator = exp_val - 1
        LocalTensor<float> minLocal2 = minQueue.AllocTensor<float>();
        Adds(minLocal2, outLocal, (float)(-1.0f), this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        
        // denominator = exp_val + 1
        Adds(outLocal, outLocal, (float)(1.0f), this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        
        // tanh = num / den
        Div(outLocal, minLocal2, outLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        minQueue.FreeTensor(minLocal2);
        
        // outLocal = 1 + tanh(...)
        Adds(outLocal, outLocal, (float)1.0f, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        // outLocal = 0.5 * x * (1 + tanh(...))
        Mul(outLocal, outLocal, sumLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        Muls(outLocal, outLocal, (float)0.5f, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        
        inQueue.FreeTensor(tmpLocal);
        sumQueue.FreeTensor(sumLocal);

        // Step 3: Add bias
        // bias shape is (1,1,1), broadcast to width
        // Load bias value - single element
        LocalTensor<float> biasLocal = inQueue.AllocTensor<float>();
        DataCopy(biasLocal, biasGm[0], ((this->biasLength + 7) / 8) * 8);
        pipe_barrier(PIPE_ALL);
        
        // Since bias is just one value, we add it as scalar
        // We need to broadcast - use Adds with scalar
        // But we loaded bias to local tensor, we need to extract the value
        // Use Adds with the bias tensor broadcasted
        // Actually for a single scalar bias, let's just use Adds approach
        // We'll read the first element
        float biasVal = biasLocal.GetValue(0);
        Adds(outLocal, outLocal, biasVal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        
        inQueue.FreeTensor(biasLocal);

        // Copy result out
        uint32_t outOffset = batchIdx * width;
        DataCopy(zGm[outOffset], outLocal, this->alignedWidth);
        pipe_barrier(PIPE_ALL);
        outQueue.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> minQueue;
    TQue<TPosition::VECIN, BUFFER_NUM> sumQueue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> zGm;
    uint32_t batchSize, channels, height, width;
    uint32_t biasLength;
    uint32_t batchPerBlock;
    uint32_t batchStart, batchEnd;
    uint32_t alignedWidth;
};

extern "C" __global__ __aicore__ void conv_transpose2d_min_sum_gelu_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinSumGeluAdd op;
    op.Init(x, bias, z,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.height, tiling_data.width, tiling_data.biasLength);
    op.Process();
}
