
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void gather_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t numRows = tiling_data.numRows;
    uint32_t srcCols = tiling_data.srcCols;
    uint32_t idxCols = tiling_data.idxCols;

    uint32_t numBlocks = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();

    // Each block processes a subset of rows
    uint32_t rowsPerBlock = (numRows + numBlocks - 1) / numBlocks;
    uint32_t rowStart = blockIdx * rowsPerBlock;
    uint32_t rowEnd = rowStart + rowsPerBlock;
    if (rowEnd > numRows) rowEnd = numRows;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int64_t> idxGm;
    AscendC::GlobalTensor<float> zGm;

    xGm.SetGlobalBuffer((__gm__ float*)x, numRows * srcCols);
    idxGm.SetGlobalBuffer((__gm__ int64_t*)idx, numRows * idxCols);
    zGm.SetGlobalBuffer((__gm__ float*)z, numRows * idxCols);

    // Process each row assigned to this block
    for (uint32_t row = rowStart; row < rowEnd; row++) {
        // Process indices in tiles
        const uint32_t TILE_SIZE = 256; // number of indices to process at a time
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;

        // We need buffers for idx (int64) and output (float)
        // int64 is 8 bytes, float is 4 bytes
        pipe.InitBuffer(idxBuf, TILE_SIZE * sizeof(int64_t));
        pipe.InitBuffer(outBuf, TILE_SIZE * sizeof(float));

        uint32_t colsProcessed = 0;
        while (colsProcessed < idxCols) {
            uint32_t curTile = TILE_SIZE;
            if (colsProcessed + curTile > idxCols) {
                curTile = idxCols - colsProcessed;
            }
            // Align curTile up to 8 for data copy (32 bytes / 4 bytes per float = 8)
            uint32_t alignedTile = ((curTile + 7) / 8) * 8;

            AscendC::LocalTensor<int64_t> idxLocal = idxBuf.Get<int64_t>();
            AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();

            // Copy index tile from global to local
            // For int64, 32 bytes = 4 elements, align to 4
            uint32_t alignedTileIdx = ((curTile + 3) / 4) * 4;
            AscendC::DataCopy(idxLocal, idxGm[row * idxCols + colsProcessed], alignedTileIdx);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Gather: for each index in the tile, read from x[row, index]
            for (uint32_t i = 0; i < curTile; i++) {
                int64_t srcIdx = idxLocal.GetValue(i);
                float val = xGm.GetValue(row * srcCols + (uint32_t)srcIdx);
                outLocal.SetValue(i, val);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            // Copy result back
            AscendC::DataCopy(zGm[row * idxCols + colsProcessed], outLocal, alignedTile);
            AscendC::PipeBarrier<PIPE_ALL>();

            colsProcessed += curTile;
        }
    }
}
