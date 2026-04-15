
#include "kernel_operator.h"

class KernelMaxPool1d {
public:
    __aicore__ inline KernelMaxPool1d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t batchSize, uint32_t channels,
                                uint32_t inputLength, uint32_t outputLength,
                                uint32_t kernelSize, uint32_t stride, uint32_t padding)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->inputLength = inputLength;
        this->outputLength = outputLength;
        this->kernelSize = kernelSize;
        this->stride = stride;
        this->padding = padding;

        this->totalRows = batchSize * channels;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->rowsPerBlock = (this->totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = this->startRow + this->rowsPerBlock;
        if (this->endRow > this->totalRows) {
            this->endRow = this->totalRows;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inputLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channels * outputLength);

        // Align tile size to 8 elements (32 bytes for float)
        uint32_t alignedInputLength = ((inputLength + 7) / 8) * 8;
        uint32_t alignedOutputLength = ((outputLength + 7) / 8) * 8;

        pipe.InitBuffer(inQueueX, 1, alignedInputLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, alignedOutputLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, alignedOutputLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = this->startRow; row < this->endRow; row++) {
            CopyIn(row);
            Compute(row);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        uint32_t alignedInputLength = ((inputLength + 7) / 8) * 8;
        // Initialize to -FLT_MAX for padding handling
        AscendC::Duplicate(xLocal, -3.402823466e+38f, alignedInputLength);
        AscendC::DataCopy(xLocal, xGm[row * inputLength], inputLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();

        uint32_t alignedOutputLength = ((outputLength + 7) / 8) * 8;

        // Initialize output to -FLT_MAX
        AscendC::Duplicate(yLocal, -3.402823466e+38f, alignedOutputLength);

        for (uint32_t k = 0; k < kernelSize; k++) {
            // For each output position o: candidate = x[o * stride - padding + k]
            // We need to gather these values
            for (uint32_t o = 0; o < outputLength; o++) {
                int32_t inIdx = (int32_t)(o * stride) - (int32_t)padding + (int32_t)k;
                float val = -3.402823466e+38f;
                if (inIdx >= 0 && inIdx < (int32_t)inputLength) {
                    val = xLocal.GetValue(inIdx);
                }
                float curMax = yLocal.GetValue(o);
                if (val > curMax) {
                    yLocal.SetValue(o, val);
                }
            }
        }

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        tmpBuf.FreeTensor(tmpLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[row * outputLength], yLocal, outputLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t channels;
    uint32_t inputLength;
    uint32_t outputLength;
    uint32_t kernelSize;
    uint32_t stride;
    uint32_t padding;
    uint32_t totalRows;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t endRow;
};

extern "C" __global__ __aicore__ void max_pooling_1d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPool1d op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.inputLength, tiling_data.outputLength,
            tiling_data.kernelSize, tiling_data.stride, tiling_data.padding);
    op.Process();
}
