
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3dMultiplyInstanceNormClampMultiplyMax {
public:
    __aicore__ inline KernelConv3dMultiplyInstanceNormClampMultiplyMax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR multiplier, GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t depth, uint32_t height, uint32_t width,
                                uint32_t kernelDepth, uint32_t kernelHeight, uint32_t kernelWidth,
                                float clampMin, float clampMax)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->depth = depth;
        this->height = height;
        this->width = width;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->clampMin = clampMin;
        this->clampMax = clampMax;

        this->totalElements = batchSize * outChannels * depth * height * width;
        this->blockLength = totalElements / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT *)weight, outChannels * inChannels * kernelDepth * kernelHeight * kernelWidth);
        if (bias != 0) {
            biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outChannels);
        } else {
            biasGm.SetGlobalBuffer(nullptr, 0);
        }
        multiplierGm.SetGlobalBuffer((__gm__ DTYPE_MULTIPLIER *)multiplier, outChannels * depth * height * width);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, this->tileLength * sizeof(DTYPE_WEIGHT));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(inQueueMultiplier, BUFFER_NUM, this->tileLength * sizeof(DTYPE_MULTIPLIER));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.AllocTensor<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_MULTIPLIER> multiplierLocal = inQueueMultiplier.AllocTensor<DTYPE_MULTIPLIER>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightLocal, weightGm[0], this->tileLength);
        if (biasGm.GetGlobalBuffer() != nullptr) {
            AscendC::DataCopy(biasLocal, biasGm[0], this->tileLength);
        } else {
            AscendC::Fill(biasLocal, 0.0f);
        }
        AscendC::DataCopy(multiplierLocal, multiplierGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueWeight.EnQue(weightLocal);
        inQueueBias.EnQue(biasLocal);
        inQueueMultiplier.EnQue(multiplierLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWeight.DeQue<DTYPE_WEIGHT>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_MULTIPLIER> multiplierLocal = inQueueMultiplier.DeQue<DTYPE_MULTIPLIER>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // Conv3d
        AscendC::Conv3d(yLocal, xLocal, weightLocal, biasLocal, this->tileLength, this->inChannels, this->outChannels, this->depth, this->height, this->width, this->kernelDepth, this->kernelHeight, this->kernelWidth);

        // Multiply
        AscendC::Mul(yLocal, yLocal, multiplierLocal, this->tileLength);

        // InstanceNorm
        AscendC::InstanceNorm(yLocal, yLocal, this->tileLength, this->outChannels, this->depth * this->height * this->width);

        // Clamp
        AscendC::Clip(yLocal, yLocal, this->clampMin, this->clampMax, this->tileLength);

        // Multiply again
        AscendC::Mul(yLocal, yLocal, multiplierLocal, this->tileLength);

        // Max operation along channel dimension
        AscendC::ReduceMax(yLocal, yLocal, this->tileLength, this->outChannels, this->depth * this->height * this->width);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueWeight.FreeTensor(weightLocal);
        inQueueBias.FreeTensor(biasLocal);
        inQueueMultiplier.FreeTensor(multiplierLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueBias, inQueueMultiplier;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT> weightGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_MULTIPLIER> multiplierGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t depth;
    uint32_t height;
    uint32_t width;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    float clampMin;
    float clampMax;
    uint32_t totalElements;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv3d_multiply_instance_norm_clamp_multiply_max_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR multiplier, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMultiplyInstanceNormClampMultiplyMax op;
    op.Init(x, weight, bias, multiplier, y,
            tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.depth, tiling_data.height, tiling_data.width,
            tiling_data.kernelDepth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.clampMin, tiling_data.clampMax);
    op.Process();
}
