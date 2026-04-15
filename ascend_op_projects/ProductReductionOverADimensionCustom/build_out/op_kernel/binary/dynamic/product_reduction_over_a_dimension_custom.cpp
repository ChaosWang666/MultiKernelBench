
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelProductReduction {
public:
    __aicore__ inline KernelProductReduction() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dim,
                                 uint32_t dimSize, uint32_t outerSize, uint32_t innerSize,
                                 uint32_t outputLength, uint32_t tileNum)
    {
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;
        this->outputLength = outputLength;

        // Each block handles a portion of the output elements
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->blockOutputStart = (outputLength / blockNum) * blockIdx;
        uint32_t remainder = outputLength % blockNum;
        if (blockIdx < remainder) {
            this->blockOutputLen = outputLength / blockNum + 1;
            this->blockOutputStart += blockIdx;
        } else {
            this->blockOutputLen = outputLength / blockNum;
            this->blockOutputStart += remainder;
        }

        if (this->blockOutputLen == 0) {
            this->blockOutputLen = 0;
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockOutputStart, this->blockOutputLen);

        // We'll process innerSize elements at a time for each (outer, dimSlice)
        // Tile size for inner dimension processing
        uint32_t maxTileSize = 1024;
        if (innerSize < maxTileSize) {
            this->tileLength = innerSize;
        } else {
            this->tileLength = maxTileSize;
        }
        // Align tileLength to 8 (32 bytes / 4 bytes per float)
        if (this->tileLength % 8 != 0 && this->tileLength >= 8) {
            this->tileLength = (this->tileLength / 8) * 8;
        }
        if (this->tileLength < 8) {
            this->tileLength = 8;
        }

        pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->blockOutputLen == 0) return;

        // Each output element at flat index `outIdx` corresponds to:
        //   outer = outIdx / innerSize
        //   inner = outIdx % innerSize
        // The product is over x[outer * dimSize * innerSize + d * innerSize + inner] for d in [0, dimSize)

        if (innerSize >= 8 && innerSize % 8 == 0) {
            // Fast path: process innerSize elements in tiles
            ProcessAligned();
        } else {
            // Scalar fallback
            ProcessScalar();
        }
    }

private:
    __aicore__ inline void ProcessAligned()
    {
        // Process output elements in tiles of tileLength along inner dimension
        for (uint32_t outIdx = 0; outIdx < this->blockOutputLen; ) {
            uint32_t globalOutIdx = this->blockOutputStart + outIdx;
            uint32_t outerIdx = globalOutIdx / innerSize;
            uint32_t innerIdx = globalOutIdx % innerSize;

            // How many contiguous inner elements can we process?
            uint32_t contiguous = innerSize - innerIdx;
            uint32_t remaining = this->blockOutputLen - outIdx;
            if (contiguous > remaining) contiguous = remaining;

            // Process in tiles
            uint32_t processed = 0;
            while (processed < contiguous) {
                uint32_t curTile = contiguous - processed;
                if (curTile > this->tileLength) curTile = this->tileLength;
                // Align to 8
                uint32_t alignedTile = (curTile + 7) / 8 * 8;

                uint32_t curInner = innerIdx + processed;
                uint32_t baseOffset = outerIdx * dimSize * innerSize + curInner;

                // Initialize accumulator with first dim slice
                AscendC::LocalTensor<float> accLocal = outQueueZ.AllocTensor<float>();
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

                AscendC::DataCopy(accLocal, xGm[baseOffset], alignedTile);
                AscendC::PipeBarrier<PIPE_ALL>();

                // Multiply with remaining dim slices
                for (uint32_t d = 1; d < dimSize; d++) {
                    uint32_t srcOffset = baseOffset + d * innerSize;
                    AscendC::DataCopy(xLocal, xGm[srcOffset], alignedTile);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Mul(accLocal, accLocal, xLocal, alignedTile);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }

                // Write output
                AscendC::DataCopy(yGm[outIdx + processed], accLocal, alignedTile);
                AscendC::PipeBarrier<PIPE_ALL>();

                inQueueX.FreeTensor(xLocal);
                outQueueZ.FreeTensor(accLocal);

                processed += curTile;
            }
            outIdx += contiguous;
        }
    }

    __aicore__ inline void ProcessScalar()
    {
        for (uint32_t outIdx = 0; outIdx < this->blockOutputLen; outIdx++) {
            uint32_t globalOutIdx = this->blockOutputStart + outIdx;
            uint32_t outerIdx = globalOutIdx / innerSize;
            uint32_t innerIdx = globalOutIdx % innerSize;

            uint32_t baseOffset = outerIdx * dimSize * innerSize + innerIdx;

            // Compute product over dimSize using vector ops on tileLength=8 aligned buffer
            AscendC::LocalTensor<float> accLocal = outQueueZ.AllocTensor<float>();
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

            // We'll use element 0 of the local tensors
            // Load first value
            AscendC::DataCopy(accLocal, xGm[baseOffset], 8);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t d = 1; d < dimSize; d++) {
                AscendC::DataCopy(xLocal, xGm[baseOffset + d * innerSize], 8);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mul(accLocal, accLocal, xLocal, 8);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            AscendC::DataCopy(yGm[outIdx], accLocal, 8);
            AscendC::PipeBarrier<PIPE_ALL>();

            outQueueZ.FreeTensor(accLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t outputLength;
    uint32_t blockOutputStart;
    uint32_t blockOutputLen;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void product_reduction_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelProductReduction op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dim,
            tiling_data.dimSize, tiling_data.outerSize, tiling_data.innerSize,
            tiling_data.outputLength, tiling_data.tileNum);
    op.Process();
}
