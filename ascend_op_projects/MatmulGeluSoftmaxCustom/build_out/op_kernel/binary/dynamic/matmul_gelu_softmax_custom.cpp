
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulGeluSoftmax {
public:
    __aicore__ inline KernelMatmulGeluSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t outFeatures, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->outFeatures = outFeatures;
        this->totalRows = batchSize;
        // Each block processes a chunk of rows
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->rowsPerBlock = (this->totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        if (this->startRow + this->rowsPerBlock > this->totalRows) {
            this->rowsPerBlock = this->totalRows > this->startRow ? this->totalRows - this->startRow : 0;
        }
        this->tileNum = tileNum;
        // Each row has outFeatures elements; tileLength is number of elements per tile within a row processing
        this->colLength = outFeatures;
        // We'll process each row, tiling the columns
        this->tileLength = (this->colLength + this->tileNum - 1) / this->tileNum;
        // Align tileLength to 32 bytes (8 floats)
        if (this->tileLength % 8 != 0) {
            this->tileLength = (this->tileLength / 8 + 1) * 8;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x + this->startRow * this->colLength, this->rowsPerBlock * this->colLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->startRow * this->colLength, this->rowsPerBlock * this->colLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, 8 * sizeof(float)); // for reduction
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < this->rowsPerBlock; row++) {
            // Pass 1: Compute GELU and find max for numerical stability
            float rowMax = -3.4e38f;
            uint32_t offset = row * this->colLength;
            uint32_t remaining = this->colLength;

            for (uint32_t t = 0; t < this->tileNum && remaining > 0; t++) {
                uint32_t curLen = this->tileLength;
                if (curLen > remaining) {
                    curLen = remaining;
                }
                // Align curLen for vector operations
                uint32_t alignedLen = curLen;
                if (alignedLen % 8 != 0) {
                    alignedLen = (alignedLen / 8 + 1) * 8;
                }

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[offset + t * this->tileLength], alignedLen);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

                // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                // Approximate GELU: x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
                // Simpler approximation: use sigmoid-based GELU = x * sigmoid(1.702 * x)
                // Even simpler: GELU ~ x * sigmoid(1.702 * x)

                AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

                // Compute 1.702 * x
                AscendC::Muls(tmp1, xIn, 1.702f, alignedLen);
                // Compute sigmoid: 1/(1+exp(-val))
                AscendC::Muls(tmp1, tmp1, -1.0f, alignedLen);
                AscendC::Exp(tmp1, tmp1, alignedLen);
                AscendC::Adds(tmp1, tmp1, 1.0f, alignedLen);
                AscendC::Reciprocal(tmp1, tmp1, alignedLen);
                // Multiply by x
                AscendC::Mul(yLocal, xIn, tmp1, alignedLen);

                // Find max in this tile
                AscendC::LocalTensor<float> redBuf = tmpBuf2.Get<float>();
                AscendC::ReduceMax(redBuf, yLocal, alignedLen, false);
                float tileMax = redBuf.GetValue(0);
                if (tileMax > rowMax) {
                    rowMax = tileMax;
                }

                // Store GELU result back to global memory temporarily
                AscendC::DataCopy(yGm[offset + t * this->tileLength], yLocal, alignedLen);
                outQueueY.FreeTensor(yLocal);
                inQueueX.FreeTensor(xIn);

                remaining -= curLen;
            }

            // Pass 2: Compute exp(gelu - max) and sum
            float rowSum = 0.0f;
            remaining = this->colLength;

            for (uint32_t t = 0; t < this->tileNum && remaining > 0; t++) {
                uint32_t curLen = this->tileLength;
                if (curLen > remaining) {
                    curLen = remaining;
                }
                uint32_t alignedLen = curLen;
                if (alignedLen % 8 != 0) {
                    alignedLen = (alignedLen / 8 + 1) * 8;
                }

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, yGm[offset + t * this->tileLength], alignedLen);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

                // Subtract max
                AscendC::Adds(yLocal, xIn, -rowMax, alignedLen);
                // Exp
                AscendC::Exp(yLocal, yLocal, alignedLen);

                // If there's padding beyond curLen, zero it out
                if (alignedLen > curLen) {
                    for (uint32_t p = curLen; p < alignedLen; p++) {
                        yLocal.SetValue(p, 0.0f);
                    }
                }

                // Sum
                AscendC::LocalTensor<float> redBuf = tmpBuf2.Get<float>();
                AscendC::ReduceSum(redBuf, yLocal, alignedLen, false);
                rowSum += redBuf.GetValue(0);

                // Store exp result
                AscendC::DataCopy(yGm[offset + t * this->tileLength], yLocal, alignedLen);
                outQueueY.FreeTensor(yLocal);
                inQueueX.FreeTensor(xIn);

                remaining -= curLen;
            }

            // Pass 3: Divide by sum
            float invSum = 1.0f / rowSum;
            remaining = this->colLength;

            for (uint32_t t = 0; t < this->tileNum && remaining > 0; t++) {
                uint32_t curLen = this->tileLength;
                if (curLen > remaining) {
                    curLen = remaining;
                }
                uint32_t alignedLen = curLen;
                if (alignedLen % 8 != 0) {
                    alignedLen = (alignedLen / 8 + 1) * 8;
                }

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, yGm[offset + t * this->tileLength], alignedLen);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

                AscendC::Muls(yLocal, xIn, invSum, alignedLen);

                AscendC::DataCopy(yGm[offset + t * this->tileLength], yLocal, alignedLen);
                outQueueY.FreeTensor(yLocal);
                inQueueX.FreeTensor(xIn);

                remaining -= curLen;
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t outFeatures;
    uint32_t totalRows;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t colLength;
};

extern "C" __global__ __aicore__ void matmul_gelu_softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulGeluSoftmax op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.outFeatures, tiling_data.tileNum);
    op.Process();
}
