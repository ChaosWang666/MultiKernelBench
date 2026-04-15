
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void scatter_add_custom(GM_ADDR x, GM_ADDR idx, GM_ADDR updates, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t numRows = tiling_data.numRows;
    uint32_t xCols = tiling_data.xCols;
    uint32_t idxCols = tiling_data.idxCols;
    
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    
    // Each block processes multiple rows
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> idxGm;
    AscendC::GlobalTensor<float> updatesGm;
    AscendC::GlobalTensor<float> zGm;
    
    xGm.SetGlobalBuffer((__gm__ float*)x, numRows * xCols);
    idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, numRows * idxCols);
    updatesGm.SetGlobalBuffer((__gm__ float*)updates, numRows * idxCols);
    zGm.SetGlobalBuffer((__gm__ float*)z, numRows * xCols);
    
    AscendC::TPipe pipe;
    
    // Align sizes to 32 bytes (8 floats)
    uint32_t xColsAligned = ((xCols + 7) / 8) * 8;
    uint32_t idxColsAligned = ((idxCols + 7) / 8) * 8;
    
    // Buffers for one row of x/z and one row of idx/updates
    AscendC::TBuf<AscendC::TPosition::VECIN> inBufX, inBufIdx, inBufUpd;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBufZ;
    
    pipe.InitBuffer(inBufX, xColsAligned * sizeof(float));
    pipe.InitBuffer(inBufIdx, idxColsAligned * sizeof(int32_t));
    pipe.InitBuffer(inBufUpd, idxColsAligned * sizeof(float));
    pipe.InitBuffer(outBufZ, xColsAligned * sizeof(float));
    
    for (uint32_t row = blockIdx; row < numRows; row += blockNum) {
        // Get local tensors
        AscendC::LocalTensor<float> xLocal = inBufX.Get<float>();
        AscendC::LocalTensor<float> zLocal = outBufZ.Get<float>();
        AscendC::LocalTensor<int32_t> idxLocal = inBufIdx.Get<int32_t>();
        AscendC::LocalTensor<float> updLocal = inBufUpd.Get<float>();
        
        // Copy x row to local
        AscendC::DataCopy(xLocal, xGm[row * xCols], xColsAligned);
        // Copy idx row to local
        AscendC::DataCopy(idxLocal, idxGm[row * idxCols], idxColsAligned);
        // Copy updates row to local
        AscendC::DataCopy(updLocal, updatesGm[row * idxCols], idxColsAligned);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Copy x to z first
        AscendC::DataCopy(zLocal, xLocal, xColsAligned);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Perform scatter add: z[idx[j]] += updates[j]
        for (uint32_t j = 0; j < idxCols; j++) {
            int32_t targetIdx = idxLocal.GetValue(j);
            float updVal = updLocal.GetValue(j);
            float curVal = zLocal.GetValue(targetIdx);
            zLocal.SetValue(targetIdx, curVal + updVal);
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Copy result back to global
        AscendC::DataCopy(zGm[row * xCols], zLocal, xColsAligned);
        
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
