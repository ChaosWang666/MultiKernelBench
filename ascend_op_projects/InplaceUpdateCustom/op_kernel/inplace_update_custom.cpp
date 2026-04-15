
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void inplace_update_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t numRows = tiling_data.numRows;
    uint32_t rowLength = tiling_data.rowLength;
    uint32_t numIdx = tiling_data.numIdx;
    uint32_t totalX = numRows * rowLength;
    uint32_t totalValue = numIdx * rowLength;
    
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> valueGm;
    AscendC::GlobalTensor<float> yGm;
    
    xGm.SetGlobalBuffer((__gm__ float*)x, totalX);
    idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, numIdx);
    valueGm.SetGlobalBuffer((__gm__ float*)value, totalValue);
    yGm.SetGlobalBuffer((__gm__ float*)y, totalX);
    
    AscendC::TPipe pipe;
    
    // First copy x to y
    // Process in tiles
    const uint32_t TILE_SIZE = 1024; // elements per tile
    uint32_t copyTiles = (totalX + TILE_SIZE - 1) / TILE_SIZE;
    
    AscendC::TQue<AscendC::TPosition::VECIN, 2> copyQueue;
    pipe.InitBuffer(copyQueue, 2, TILE_SIZE * sizeof(float));
    
    for (uint32_t t = 0; t < copyTiles; t++) {
        uint32_t offset = t * TILE_SIZE;
        uint32_t len = TILE_SIZE;
        if (offset + len > totalX) {
            len = totalX - offset;
        }
        // Align len to 8 for DataCopy (32 bytes = 8 floats)
        uint32_t alignedLen = (len + 7) / 8 * 8;
        if (offset + alignedLen > totalX) {
            alignedLen = len; // fallback
        }
        
        AscendC::LocalTensor<float> tmpLocal = copyQueue.AllocTensor<float>();
        AscendC::DataCopy(tmpLocal, xGm[offset], alignedLen);
        copyQueue.EnQue(tmpLocal);
        
        AscendC::LocalTensor<float> tmpOut = copyQueue.DeQue<float>();
        AscendC::DataCopy(yGm[offset], tmpOut, alignedLen);
        copyQueue.FreeTensor(tmpOut);
    }
    
    // Now for each index, copy value row to y
    // We need to read idx values - copy them to local buffer
    // Process idx in small batches
    const uint32_t IDX_BATCH = 256;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> idxQueue;
    pipe.InitBuffer(idxQueue, 1, IDX_BATCH * sizeof(int32_t));
    
    AscendC::TQue<AscendC::TPosition::VECIN, 2> valQueue;
    pipe.InitBuffer(valQueue, 2, rowLength * sizeof(float));
    
    for (uint32_t ib = 0; ib < numIdx; ib += IDX_BATCH) {
        uint32_t batchSize = IDX_BATCH;
        if (ib + batchSize > numIdx) {
            batchSize = numIdx - ib;
        }
        uint32_t alignedBatch = (batchSize + 7) / 8 * 8;
        
        // Copy index batch
        AscendC::LocalTensor<int32_t> idxLocal = idxQueue.AllocTensor<int32_t>();
        AscendC::DataCopy(idxLocal, idxGm[ib], alignedBatch);
        idxQueue.EnQue(idxLocal);
        AscendC::LocalTensor<int32_t> idxData = idxQueue.DeQue<int32_t>();
        
        for (uint32_t j = 0; j < batchSize; j++) {
            int32_t rowIdx = idxData.GetValue(j);
            uint32_t srcOffset = (ib + j) * rowLength;
            uint32_t dstOffset = (uint32_t)rowIdx * rowLength;
            
            // Copy value row to y at the target row
            uint32_t alignedRowLen = (rowLength + 7) / 8 * 8;
            
            AscendC::LocalTensor<float> vLocal = valQueue.AllocTensor<float>();
            AscendC::DataCopy(vLocal, valueGm[srcOffset], alignedRowLen);
            valQueue.EnQue(vLocal);
            
            AscendC::LocalTensor<float> vOut = valQueue.DeQue<float>();
            AscendC::DataCopy(yGm[dstOffset], vOut, alignedRowLen);
            valQueue.FreeTensor(vOut);
        }
        
        idxQueue.FreeTensor(idxData);
    }
}
