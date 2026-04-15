
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void scatter_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR updates, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t numRows = tiling_data.numRows;
    uint32_t xCols = tiling_data.xCols;
    uint32_t idxCols = tiling_data.idxCols;

    uint32_t numBlocks = AscendC::GetBlockNum();
    uint32_t blockIdx = AscendC::GetBlockIdx();

    // Each block processes a range of rows
    uint32_t rowsPerBlock = (numRows + numBlocks - 1) / numBlocks;
    uint32_t rowStart = blockIdx * rowsPerBlock;
    uint32_t rowEnd = rowStart + rowsPerBlock;
    if (rowEnd > numRows) rowEnd = numRows;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> updGm;
    AscendC::GlobalTensor<float> zGm;

    xGm.SetGlobalBuffer((__gm__ float*)x, numRows * xCols);
    idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, numRows * idxCols);
    updGm.SetGlobalBuffer((__gm__ float*)updates, numRows * idxCols);
    zGm.SetGlobalBuffer((__gm__ float*)z, numRows * xCols);

    AscendC::TPipe pipe;

    // We'll process row by row for this block's assigned rows
    // Use tiled approach for copying x to z, then scatter updates

    // Align tile sizes to 32 bytes = 8 floats
    const uint32_t FLOAT_ALIGN = 8;
    const uint32_t INT_ALIGN = 8;

    // Buffer size for x copy tiles
    uint32_t xTileSize = 2048; // floats per tile for x copy
    if (xTileSize > xCols) xTileSize = xCols;
    // Align down
    xTileSize = (xTileSize / FLOAT_ALIGN) * FLOAT_ALIGN;

    uint32_t idxTileSize = 2048;
    if (idxTileSize > idxCols) idxTileSize = idxCols;
    idxTileSize = (idxTileSize / INT_ALIGN) * INT_ALIGN;

    // Allocate buffers
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueIdx;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueUpd;

    pipe.InitBuffer(inQueueX, 1, xTileSize * sizeof(float));
    pipe.InitBuffer(outQueueZ, 1, xTileSize * sizeof(float));
    pipe.InitBuffer(inQueueIdx, 1, idxTileSize * sizeof(int32_t));
    pipe.InitBuffer(inQueueUpd, 1, idxTileSize * sizeof(float));

    for (uint32_t row = rowStart; row < rowEnd; row++) {
        // Step 1: Copy x row to z row
        uint32_t xOffset = row * xCols;
        uint32_t numXTiles = (xCols + xTileSize - 1) / xTileSize;
        for (uint32_t t = 0; t < numXTiles; t++) {
            uint32_t tStart = t * xTileSize;
            uint32_t tLen = xTileSize;
            if (tStart + tLen > xCols) tLen = xCols - tStart;
            // Align tLen up to FLOAT_ALIGN for DataCopy
            uint32_t copyLen = ((tLen + FLOAT_ALIGN - 1) / FLOAT_ALIGN) * FLOAT_ALIGN;

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[xOffset + tStart], copyLen);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xDq = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            AscendC::DataCopy(zLocal, xDq, copyLen);
            outQueueZ.EnQue(zLocal);
            inQueueX.FreeTensor(xDq);

            AscendC::LocalTensor<float> zDq = outQueueZ.DeQue<float>();
            AscendC::DataCopy(zGm[xOffset + tStart], zDq, copyLen);
            outQueueZ.FreeTensor(zDq);
        }

        // Step 2: Apply scatter - process idx/updates in tiles
        // We need to read idx and updates, then for each element do z[row][idx[i]] = updates[i]
        // This requires scalar writes to global memory
        uint32_t idxOffset = row * idxCols;
        uint32_t numIdxTiles = (idxCols + idxTileSize - 1) / idxTileSize;

        for (uint32_t t = 0; t < numIdxTiles; t++) {
            uint32_t tStart = t * idxTileSize;
            uint32_t tLen = idxTileSize;
            if (tStart + tLen > idxCols) tLen = idxCols - tStart;
            uint32_t copyLenIdx = ((tLen + INT_ALIGN - 1) / INT_ALIGN) * INT_ALIGN;
            uint32_t copyLenUpd = ((tLen + FLOAT_ALIGN - 1) / FLOAT_ALIGN) * FLOAT_ALIGN;

            AscendC::LocalTensor<int32_t> idxLocal = inQueueIdx.AllocTensor<int32_t>();
            AscendC::DataCopy(idxLocal, idxGm[idxOffset + tStart], copyLenIdx);
            inQueueIdx.EnQue(idxLocal);

            AscendC::LocalTensor<float> updLocal = inQueueUpd.AllocTensor<float>();
            AscendC::DataCopy(updLocal, updGm[idxOffset + tStart], copyLenUpd);
            inQueueUpd.EnQue(updLocal);

            AscendC::LocalTensor<int32_t> idxDq = inQueueIdx.DeQue<int32_t>();
            AscendC::LocalTensor<float> updDq = inQueueUpd.DeQue<float>();

            // Scatter: write each update to the correct position in z
            for (uint32_t i = 0; i < tLen; i++) {
                int32_t col = idxDq.GetValue(i);
                float val = updDq.GetValue(i);
                zGm.SetValue(xOffset + col, val);
            }

            inQueueIdx.FreeTensor(idxDq);
            inQueueUpd.FreeTensor(updDq);
        }
    }
}
