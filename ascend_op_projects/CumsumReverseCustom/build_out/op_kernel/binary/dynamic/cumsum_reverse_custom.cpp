
#include "kernel_operator.h"

// Reverse cumulative sum along a specified dimension
// For 2D tensor with dim=1 and innerSize=1:
//   Each row of length dimSize needs reverse cumsum
//   reverse cumsum: y[i] = sum(x[j] for j in [i, dimSize-1])
//   Equivalently: flip, cumsum, flip

// We distribute rows (outer indices) across blocks and process them.
// Each block handles a subset of rows. For each row, we do the reverse cumsum sequentially.

extern "C" __global__ __aicore__ void cumsum_reverse_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t outerSize = tiling_data.outerSize;
    uint32_t dimSize = tiling_data.dimSize;
    uint32_t innerSize = tiling_data.innerSize;

    // Total number of "lines" to process
    // Each line is defined by (outer_idx, inner_idx) and has length dimSize along the target dim
    uint32_t totalLines = outerSize * innerSize;

    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();

    // Distribute lines across blocks
    uint32_t linesPerBlock = (totalLines + blockNum - 1) / blockNum;
    uint32_t lineStart = blockIdx * linesPerBlock;
    uint32_t lineEnd = lineStart + linesPerBlock;
    if (lineEnd > totalLines) lineEnd = totalLines;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, outerSize * dimSize * innerSize);
    yGm.SetGlobalBuffer((__gm__ float*)y, outerSize * dimSize * innerSize);

    // For innerSize == 1 (our specific case: dim=1 on 2D tensor),
    // each line is a contiguous row of length dimSize.
    // We process using vector operations with tiling on UB.

    AscendC::TPipe pipe;

    // We'll use a simple approach: allocate UB buffers and process tile by tile
    // For each row, we copy the row into UB, compute reverse cumsum, write back.

    // Max UB we can use per buffer (conservative estimate)
    // We'll process one row at a time if it fits, or tile within a row.

    // dimSize = 32768 floats = 128KB, UB is typically 256KB on 910B
    // We can fit one row in UB.

    const uint32_t TILE_SIZE = 8192; // process in tiles of this many elements
    const uint32_t BUFFER_NUM = 1;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;

    if (innerSize == 1) {
        // Contiguous rows case
        // For reverse cumsum we need to process from end to start
        // Strategy: load tiles from end to start, maintain running sum

        // We'll allocate a single tile buffer for loading and a single for output
        // Process each row from right to left in tiles of TILE_SIZE

        uint32_t tileSizeBytes = TILE_SIZE * sizeof(float);
        pipe.InitBuffer(inQueue, BUFFER_NUM, tileSizeBytes);
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileSizeBytes);

        for (uint32_t line = lineStart; line < lineEnd; line++) {
            uint32_t rowOffset = line * dimSize;
            float runningSum = 0.0f;

            // Number of full tiles
            uint32_t numFullTiles = dimSize / TILE_SIZE;
            uint32_t remainder = dimSize % TILE_SIZE;

            // Process from the end of the row to the beginning
            // Tile boundaries (from end): 
            // Last tile starts at (numFullTiles * TILE_SIZE) if remainder > 0, else ((numFullTiles-1)*TILE_SIZE)

            // Process remainder tile first (rightmost part)
            if (remainder > 0) {
                uint32_t tileOffset = numFullTiles * TILE_SIZE;
                // We need remainder to be aligned to 32 bytes (8 floats) for DataCopy
                // If not aligned, we pad
                uint32_t alignedLen = ((remainder + 7) / 8) * 8;

                AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

                AscendC::DataCopy(inLocal, xGm[rowOffset + tileOffset], alignedLen);
                inQueue.EnQue(inLocal);
                inLocal = inQueue.DeQue<float>();

                // Reverse cumsum within this tile (from index remainder-1 down to 0)
                // Then add runningSum to all
                // Since this is the rightmost tile, runningSum starts at 0
                float tileSum = 0.0f;
                for (int32_t i = (int32_t)remainder - 1; i >= 0; i--) {
                    tileSum += inLocal.GetValue(i);
                    outLocal.SetValue(i, tileSum);
                }
                runningSum = tileSum;

                // Zero out padding
                for (uint32_t i = remainder; i < alignedLen; i++) {
                    outLocal.SetValue(i, 0.0f);
                }

                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[rowOffset + tileOffset], outLocal, alignedLen);

                inQueue.FreeTensor(inLocal);
                outQueue.FreeTensor(outLocal);
            }

            // Process full tiles from right to left
            for (int32_t t = (int32_t)numFullTiles - 1; t >= 0; t--) {
                uint32_t tileOffset = (uint32_t)t * TILE_SIZE;

                AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

                AscendC::DataCopy(inLocal, xGm[rowOffset + tileOffset], TILE_SIZE);
                inQueue.EnQue(inLocal);
                inLocal = inQueue.DeQue<float>();

                // Reverse cumsum within tile, then add runningSum
                float tileSum = 0.0f;
                for (int32_t i = (int32_t)TILE_SIZE - 1; i >= 0; i--) {
                    tileSum += inLocal.GetValue(i);
                    outLocal.SetValue(i, tileSum + runningSum);
                }
                runningSum += tileSum;

                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<float>();
                AscendC::DataCopy(yGm[rowOffset + tileOffset], outLocal, TILE_SIZE);

                inQueue.FreeTensor(inLocal);
                outQueue.FreeTensor(outLocal);
            }
        }
    } else {
        // General case with innerSize > 1 (strided access)
        // For simplicity, handle element by element with stride
        uint32_t stride = innerSize;
        uint32_t alignedDim = ((dimSize + 7) / 8) * 8;
        uint32_t bufSize = alignedDim * sizeof(float);
        if (bufSize < 32) bufSize = 32;

        pipe.InitBuffer(inQueue, BUFFER_NUM, bufSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, bufSize);

        for (uint32_t line = lineStart; line < lineEnd; line++) {
            uint32_t outerIdx = line / innerSize;
            uint32_t innerIdx = line % innerSize;
            uint32_t baseOffset = outerIdx * dimSize * innerSize + innerIdx;

            AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

            // Gather elements along dim
            for (uint32_t d = 0; d < dimSize; d++) {
                float val = xGm.GetValue(baseOffset + d * stride);
                inLocal.SetValue(d, val);
            }

            // Reverse cumsum
            float sum = 0.0f;
            for (int32_t d = (int32_t)dimSize - 1; d >= 0; d--) {
                sum += inLocal.GetValue(d);
                outLocal.SetValue(d, sum);
            }

            // Scatter back
            for (uint32_t d = 0; d < dimSize; d++) {
                yGm.SetValue(baseOffset + d * stride, outLocal.GetValue(d));
            }

            inQueue.FreeTensor(inLocal);
            outQueue.FreeTensor(outLocal);
        }
    }
}
