
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelCumsum {
public:
    __aicore__ inline KernelCumsum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dim,
                                 uint32_t dimSize, uint32_t outerSize, uint32_t innerSize)
    {
        this->totalLength = totalLength;
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;

        // Each row along the cumsum dimension has innerSize elements
        // Total number of "rows" to process = outerSize * innerSize
        // We distribute these rows across blocks
        uint32_t totalRows = outerSize * innerSize;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        if (this->startRow > totalRows) this->startRow = totalRows;
        uint32_t endRow = this->startRow + this->rowsPerBlock;
        if (endRow > totalRows) endRow = totalRows;
        this->actualRows = endRow - this->startRow;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);

        // We process one row at a time along the cumsum dim
        // Each step along the dim has innerSize elements
        // Align tile length to 8 (32 bytes / 4 bytes per float)
        this->tileLength = ((innerSize + 7) / 8) * 8;

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(accumBuf, 1, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->actualRows == 0) return;

        if (this->innerSize == 1) {
            // Special optimized path: cumsum along contiguous dimension
            // Each "row" is a sequence of dimSize contiguous floats
            ProcessContiguous();
        } else {
            ProcessGeneral();
        }
    }

private:
    __aicore__ inline void ProcessContiguous()
    {
        // innerSize == 1, so each row is dimSize contiguous elements
        // startRow is the index into outerSize dimension
        for (uint32_t r = 0; r < this->actualRows; r++) {
            uint32_t rowIdx = this->startRow + r;
            // outerIdx = rowIdx (since innerSize=1)
            uint32_t baseOffset = rowIdx * this->dimSize;

            // Process dimSize elements with tiling
            // We'll do a sequential scan with small tiles
            uint32_t alignedTile = 256; // process 256 floats at a time
            if (alignedTile > this->dimSize) {
                alignedTile = ((this->dimSize + 7) / 8) * 8;
            }

            // Use accumBuf to hold running sum (single element replicated)
            AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();
            // Initialize accumulator to 0
            AscendC::Duplicate(accumLocal, (float)0.0, alignedTile);

            uint32_t processed = 0;
            while (processed < this->dimSize) {
                uint32_t remaining = this->dimSize - processed;
                uint32_t curTile = remaining < alignedTile ? remaining : alignedTile;
                uint32_t curTileAligned = ((curTile + 7) / 8) * 8;

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

                AscendC::DataCopy(xLocal, xGm[baseOffset + processed], curTileAligned);
                inQueue.EnQue(xLocal);

                xLocal = inQueue.DeQue<float>();

                // Sequential cumsum within this tile
                // First element gets added to last accumulator value
                // We need scalar sequential scan for correctness
                float runningSum;
                // Get the last accumulated value
                if (processed == 0) {
                    runningSum = 0.0f;
                } else {
                    // runningSum was saved
                    runningSum = this->savedSum;
                }

                for (uint32_t i = 0; i < curTile; i++) {
                    runningSum += xLocal.GetValue(i);
                    yLocal.SetValue(i, runningSum);
                }
                this->savedSum = runningSum;

                outQueue.EnQue(yLocal);
                yLocal = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[baseOffset + processed], yLocal, curTileAligned);

                inQueue.FreeTensor(xLocal);
                outQueue.FreeTensor(yLocal);

                processed += curTile;
            }
        }
    }

    __aicore__ inline void ProcessGeneral()
    {
        for (uint32_t r = 0; r < this->actualRows; r++) {
            uint32_t rowIdx = this->startRow + r;
            uint32_t outerIdx = rowIdx / this->innerSize;
            uint32_t innerIdx = rowIdx % this->innerSize;

            AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();
            AscendC::Duplicate(accumLocal, (float)0.0, this->tileLength);

            for (uint32_t d = 0; d < this->dimSize; d++) {
                uint32_t offset = (outerIdx * this->dimSize + d) * this->innerSize + innerIdx;

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

                // Copy one element
                AscendC::DataCopy(xLocal, xGm[offset], this->tileLength);
                inQueue.EnQue(xLocal);
                xLocal = inQueue.DeQue<float>();

                float val = xLocal.GetValue(0);
                float acc = accumLocal.GetValue(0) + val;
                accumLocal.SetValue(0, acc);
                yLocal.SetValue(0, acc);

                outQueue.EnQue(yLocal);
                yLocal = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[offset], yLocal, this->tileLength);

                inQueue.FreeTensor(xLocal);
                outQueue.FreeTensor(yLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalLength;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t actualRows;
    uint32_t tileLength;
    float savedSum;
};

extern "C" __global__ __aicore__ void cumsum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumsum op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dim,
            tiling_data.dimSize, tiling_data.outerSize, tiling_data.innerSize);
    op.Process();
}
