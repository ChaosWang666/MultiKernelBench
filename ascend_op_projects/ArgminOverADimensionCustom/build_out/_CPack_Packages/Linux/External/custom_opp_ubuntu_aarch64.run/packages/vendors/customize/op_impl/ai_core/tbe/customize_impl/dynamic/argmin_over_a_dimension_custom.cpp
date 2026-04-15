
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void argmin_over_a_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t dimBefore = tiling_data.dimBefore;
    uint32_t dimSize = tiling_data.dimSize;
    uint32_t dimAfter = tiling_data.dimAfter;
    uint32_t numOutputElements = dimBefore * dimAfter;

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t numBlocks = AscendC::GetBlockNum();

    // Each block processes a portion of the output elements
    uint32_t elementsPerBlock = (numOutputElements + numBlocks - 1) / numBlocks;
    uint32_t startElem = blockIdx * elementsPerBlock;
    uint32_t endElem = startElem + elementsPerBlock;
    if (endElem > numOutputElements) endElem = numOutputElements;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int64_t> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, dimBefore * dimSize * dimAfter);
    yGm.SetGlobalBuffer((__gm__ int64_t*)y, numOutputElements);

    // Process each output element assigned to this block
    // For each output element (b, a) where b is index in dimBefore and a is index in dimAfter,
    // we need to find argmin over dimSize elements at positions x[b * dimSize * dimAfter + k * dimAfter + a]
    
    // We'll use a tile-based approach with local memory
    AscendC::TPipe pipe;
    
    // Allocate buffers for loading slices along the reduction dimension
    // We process in tiles along dimSize
    constexpr uint32_t BUFFER_NUM = 1;
    constexpr uint32_t TILE_SIZE = 256; // number of elements per tile (must be multiple of 32 bytes / 4 = 8 floats min)

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;

    // For the case where dimAfter == 1 (reducing along last or middle dim with contiguous access),
    // we can load contiguous chunks. For dimAfter > 1, we need strided access.

    if (dimAfter == 1) {
        // Contiguous case: for each b in [startElem, endElem), 
        // the data is at xGm[b * dimSize ... b * dimSize + dimSize - 1]
        // We tile along dimSize
        uint32_t alignedTileSize = ((TILE_SIZE + 7) / 8) * 8; // align to 8 floats (32 bytes)
        if (alignedTileSize < 8) alignedTileSize = 8;
        
        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedTileSize * sizeof(float));

        for (uint32_t elem = startElem; elem < endElem; elem++) {
            uint32_t baseOffset = elem * dimSize;
            float minVal = 3.4028235e+38f;
            int64_t minIdx = 0;

            uint32_t processed = 0;
            while (processed < dimSize) {
                uint32_t remaining = dimSize - processed;
                uint32_t curTile = remaining < alignedTileSize ? remaining : alignedTileSize;
                // Align curTile up to 8 for DataCopy
                uint32_t copyLen = ((curTile + 7) / 8) * 8;
                if (baseOffset + processed + copyLen > dimBefore * dimSize * dimAfter + copyLen) {
                    // safety, shouldn't happen for aligned total
                }

                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                // Need to be careful: copyLen might read past the end of valid data
                // We'll copy copyLen elements and only examine curTile of them
                AscendC::DataCopy(xLocal, xGm[baseOffset + processed], copyLen);
                inQueue.EnQue(xLocal);
                
                AscendC::LocalTensor<float> xProc = inQueue.DeQue<float>();
                for (uint32_t i = 0; i < curTile; i++) {
                    float val = xProc.GetValue(i);
                    if (val < minVal) {
                        minVal = val;
                        minIdx = (int64_t)(processed + i);
                    }
                }
                inQueue.FreeTensor(xProc);
                processed += curTile;
            }
            // Write result - we need to handle int64_t output
            // Use a simple GM write
            yGm.SetValue(elem, minIdx);
        }
    } else {
        // General case: dimAfter > 1
        // For each output element, b = elem / dimAfter, a = elem % dimAfter
        // Data at x[b * dimSize * dimAfter + k * dimAfter + a] for k in [0, dimSize)
        // This is strided access with stride = dimAfter

        // We'll process multiple output elements that share the same 'b' together
        // when possible, loading full rows

        // Simple approach: load tiles of the reduction dimension with gathering
        uint32_t alignedTileSize = ((TILE_SIZE + 7) / 8) * 8;
        if (alignedTileSize < 8) alignedTileSize = 8;
        
        // For strided access, we load one row at a time (dimAfter elements) and process
        // all 'a' values for a given 'b' and 'k' at once
        // A row is dimAfter elements at offset b * dimSize * dimAfter + k * dimAfter
        
        uint32_t rowSize = dimAfter;
        uint32_t alignedRowSize = ((rowSize + 7) / 8) * 8;
        
        pipe.InitBuffer(inQueue, BUFFER_NUM, alignedRowSize * sizeof(float));
        
        for (uint32_t elem = startElem; elem < endElem; elem++) {
            uint32_t b = elem / dimAfter;
            uint32_t a = elem % dimAfter;
            uint32_t baseOffset = b * dimSize * dimAfter;
            
            float minVal = 3.4028235e+38f;
            int64_t minIdx = 0;

            for (uint32_t k = 0; k < dimSize; k++) {
                // We need value at baseOffset + k * dimAfter + a
                // Load a row of dimAfter elements starting at baseOffset + k * dimAfter
                AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[baseOffset + k * dimAfter], alignedRowSize);
                inQueue.EnQue(xLocal);
                
                AscendC::LocalTensor<float> xProc = inQueue.DeQue<float>();
                float val = xProc.GetValue(a);
                if (val < minVal) {
                    minVal = val;
                    minIdx = (int64_t)k;
                }
                inQueue.FreeTensor(xProc);
            }
            yGm.SetValue(elem, minIdx);
        }
    }
}
