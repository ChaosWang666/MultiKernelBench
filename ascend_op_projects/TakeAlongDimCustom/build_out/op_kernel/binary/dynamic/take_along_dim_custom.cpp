
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void take_along_dim_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t numRows = tiling_data.numRows;
    uint32_t srcCols = tiling_data.srcCols;
    uint32_t idxCols = tiling_data.idxCols;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();

    // Each block processes a subset of rows
    __gm__ float* xGm = (__gm__ float*)x;
    __gm__ int64_t* idxGm = (__gm__ int64_t*)idx;
    __gm__ float* zGm = (__gm__ float*)z;

    for (uint32_t row = blockIdx; row < numRows; row += blockNum) {
        __gm__ float* xRow = xGm + (uint64_t)row * srcCols;
        __gm__ int64_t* idxRow = idxGm + (uint64_t)row * idxCols;
        __gm__ float* zRow = zGm + (uint64_t)row * idxCols;

        // Process elements in tiles to use local memory efficiently
        // We'll do scalar gather since indices are arbitrary
        const uint32_t TILE_SIZE = 256;
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::TPosition::VECIN> idxBuf;
        AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;

        // Allocate buffers
        pipe.InitBuffer(idxBuf, TILE_SIZE * sizeof(int64_t));
        pipe.InitBuffer(outBuf, TILE_SIZE * sizeof(float));

        uint32_t remaining = idxCols;
        uint32_t offset = 0;

        while (remaining > 0) {
            uint32_t curTile = remaining > TILE_SIZE ? TILE_SIZE : remaining;
            // Align to 32 bytes for DataCopy: int64_t is 8 bytes, so align to 4 elements
            // float is 4 bytes, align to 8 elements
            uint32_t idxAligned = ((curTile * sizeof(int64_t) + 31) / 32) * 32 / sizeof(int64_t);
            if (idxAligned < curTile) idxAligned = curTile;
            uint32_t outAligned = ((curTile * sizeof(float) + 31) / 32) * 32 / sizeof(float);
            if (outAligned < curTile) outAligned = curTile;

            AscendC::LocalTensor<int64_t> idxLocal = idxBuf.Get<int64_t>();
            AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();

            // Copy index tile
            AscendC::DataCopy(idxLocal, ((__gm__ int64_t*)(idxRow + offset)), idxAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Gather: for each index, read from xRow
            for (uint32_t i = 0; i < curTile; i++) {
                int64_t index = idxLocal.GetValue(i);
                float val = *(xRow + index);
                outLocal.SetValue(i, val);
            }

            AscendC::PipeBarrier<PIPE_ALL>();

            // Copy out
            AscendC::DataCopy(((__gm__ float*)(zRow + offset)), outLocal, outAligned);
            AscendC::PipeBarrier<PIPE_ALL>();

            offset += curTile;
            remaining -= curTile;
        }
    }
}
