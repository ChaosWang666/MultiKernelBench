
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelL2Norm {
public:
    __aicore__ inline KernelL2Norm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t dim, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->dim = dim;
        this->tileNum = tileNum;

        // Each block processes a subset of rows
        uint32_t totalRows = batchSize;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->rowsPerBlock = totalRows / blockNum;
        uint32_t remainRows = totalRows % blockNum;
        if (blockIdx < remainRows) {
            this->rowsPerBlock += 1;
            this->startRow = blockIdx * this->rowsPerBlock;
        } else {
            this->startRow = remainRows * (this->rowsPerBlock + 1) + (blockIdx - remainRows) * this->rowsPerBlock;
        }

        // Align dim to 8 for float (32 bytes = 8 floats)
        this->dimAligned = (dim + 7) / 8 * 8;

        // Tile the dim dimension
        // We process each row in tiles along the dim dimension
        this->tileLenAligned = (this->dimAligned + tileNum - 1) / tileNum;
        // Make sure tileLenAligned is aligned to 8
        this->tileLenAligned = (this->tileLenAligned + 7) / 8 * 8;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * (uint64_t)dim);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * (uint64_t)dim);

        // We need buffers for: input tile, squared tile, output tile, and a reduce sum workspace
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLenAligned * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLenAligned * sizeof(float));
        pipe.InitBuffer(workBuf, 1, this->tileLenAligned * sizeof(float));
        // Buffer for partial sum reduction - need at least dimAligned/8 + some extra
        pipe.InitBuffer(reduceBuf, 1, this->tileLenAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            ProcessRow(this->startRow + r);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        // Step 1: Compute sum of squares for this row
        float normSq = 0.0f;
        uint64_t rowOffset = (uint64_t)row * this->dim;
        uint32_t remaining = this->dim;
        uint32_t offset = 0;

        while (remaining > 0) {
            uint32_t curLen = remaining < this->tileLenAligned ? remaining : this->tileLenAligned;
            uint32_t curLenAligned = (curLen + 7) / 8 * 8;

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::LocalTensor<float> sqLocal = workBuf.AllocTensor<float>();

            // Copy input tile
            AscendC::DataCopy(xLocal, xGm[rowOffset + offset], curLenAligned);
            // Zero out padding if needed
            if (curLen < curLenAligned) {
                for (uint32_t i = curLen; i < curLenAligned; i++) {
                    xLocal.SetValue(i, 0.0f);
                }
            }

            // Square
            AscendC::Mul(sqLocal, xLocal, xLocal, curLenAligned);

            // Reduce sum
            float partialSum = 0.0f;
            AscendC::LocalTensor<float> redLocal = reduceBuf.AllocTensor<float>();

            // Use ReduceSum
            AscendC::ReduceSum(redLocal, sqLocal, workBuf, curLenAligned);
            partialSum = redLocal.GetValue(0);

            normSq += partialSum;

            reduceBuf.FreeTensor(redLocal);
            workBuf.FreeTensor(sqLocal);
            inQueueX.FreeTensor(xLocal);

            offset += curLen;
            remaining -= curLen;
        }

        // Step 2: Compute inverse norm
        float norm = 1.0f;
        if (normSq > 0.0f) {
            // sqrt
            float normVal = normSq;
            // Simple Newton's method for sqrt since we can't use math.h easily
            // Actually we can just compute on scalar
            // Use a scalar approach
            normVal = normSq;
            // Approximate sqrt using built-in
            // We'll use the vector Sqrt on a small buffer
            {
                AscendC::LocalTensor<float> tmpLocal = workBuf.AllocTensor<float>();
                tmpLocal.SetValue(0, normSq);
                // Fill 8 elements for alignment
                for (uint32_t i = 1; i < 8; i++) {
                    tmpLocal.SetValue(i, 1.0f);
                }
                AscendC::Sqrt(tmpLocal, tmpLocal, 8);
                norm = tmpLocal.GetValue(0);
                workBuf.FreeTensor(tmpLocal);
            }
        }

        float invNorm = 1.0f / norm;

        // Step 3: Normalize - multiply each element by invNorm
        remaining = this->dim;
        offset = 0;

        while (remaining > 0) {
            uint32_t curLen = remaining < this->tileLenAligned ? remaining : this->tileLenAligned;
            uint32_t curLenAligned = (curLen + 7) / 8 * 8;

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            AscendC::DataCopy(xLocal, xGm[rowOffset + offset], curLenAligned);

            // Multiply by invNorm (scalar)
            AscendC::Muls(yLocal, xLocal, invNorm, curLenAligned);

            // Copy out
            AscendC::DataCopy(yGm[rowOffset + offset], yLocal, curLenAligned);

            outQueueY.FreeTensor(yLocal);
            inQueueX.FreeTensor(xLocal);

            offset += curLen;
            remaining -= curLen;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t dim;
    uint32_t dimAligned;
    uint32_t tileNum;
    uint32_t tileLenAligned;
    uint32_t rowsPerBlock;
    uint32_t startRow;
};

extern "C" __global__ __aicore__ void l2_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelL2Norm op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.dim, tiling_data.tileNum);
    op.Process();
}
