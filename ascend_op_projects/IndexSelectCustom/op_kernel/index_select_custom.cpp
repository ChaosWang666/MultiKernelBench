
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void index_select_custom(GM_ADDR x, GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t dim = tiling_data.dim;
    uint32_t numRows = tiling_data.numRows;
    uint32_t numCols = tiling_data.numCols;
    uint32_t numIndices = tiling_data.numIndices;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t numBlocks = AscendC::GetBlockNum();

    // For dim==1: x is [numRows, numCols], output is [numRows, numIndices]
    // Each block processes a subset of rows
    // For dim==0: x is [numRows, numCols], output is [numIndices, numCols]
    // Each block processes a subset of indices

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<float> yGm;

    xGm.SetGlobalBuffer((__gm__ float*)x, numRows * numCols);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices, numIndices);
    yGm.SetGlobalBuffer((__gm__ float*)y, numRows * numIndices);

    if (dim == 1) {
        // Output shape: [numRows, numIndices]
        // Total work items = numRows * numIndices
        // We distribute rows across blocks
        uint32_t rowsPerBlock = (numRows + numBlocks - 1) / numBlocks;
        uint32_t rowStart = blockIdx * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > numRows) rowEnd = numRows;

        // We need to read indices into local memory
        // Process each row assigned to this block
        for (uint32_t row = rowStart; row < rowEnd; row++) {
            for (uint32_t j = 0; j < numIndices; j++) {
                int32_t idx = indicesGm.GetValue(j);
                float val = xGm.GetValue(row * numCols + idx);
                yGm.SetValue(row * numIndices + j, val);
            }
        }
    } else {
        // dim == 0
        // Output shape: [numIndices, numCols]
        uint32_t idxPerBlock = (numIndices + numBlocks - 1) / numBlocks;
        uint32_t idxStart = blockIdx * idxPerBlock;
        uint32_t idxEnd = idxStart + idxPerBlock;
        if (idxEnd > numIndices) idxEnd = numIndices;

        for (uint32_t i = idxStart; i < idxEnd; i++) {
            int32_t srcRow = indicesGm.GetValue(i);
            for (uint32_t c = 0; c < numCols; c++) {
                float val = xGm.GetValue(srcRow * numCols + c);
                yGm.SetValue(i * numCols + c, val);
            }
        }
    }
}
