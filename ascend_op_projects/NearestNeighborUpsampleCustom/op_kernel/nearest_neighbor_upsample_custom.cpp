
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void nearest_neighbor_upsample_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t batchSize = tiling_data.batchSize;
    uint32_t channels = tiling_data.channels;
    uint32_t inputHeight = tiling_data.inputHeight;
    uint32_t inputWidth = tiling_data.inputWidth;
    uint32_t outputHeight = tiling_data.outputHeight;
    uint32_t outputWidth = tiling_data.outputWidth;
    uint32_t scaleFactor = tiling_data.scaleFactor;
    
    uint32_t totalInputRows = batchSize * channels * inputHeight;
    uint32_t totalOutputRows = batchSize * channels * outputHeight;
    
    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    
    // Each block processes a subset of output rows
    // For each output row, we copy the corresponding input row and replicate pixels
    
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * inputHeight * inputWidth);
    yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * channels * outputHeight * outputWidth);
    
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> inBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outBuf;
    
    // Each input row is inputWidth floats; each output row is outputWidth = inputWidth * scaleFactor floats
    // We need buffer sizes aligned to 32 bytes (8 floats)
    uint32_t inputRowBytes = inputWidth * sizeof(float);
    uint32_t outputRowBytes = outputWidth * sizeof(float);
    
    // Align to 32 bytes
    uint32_t inputRowBytesAligned = (inputRowBytes + 31) & ~31;
    uint32_t outputRowBytesAligned = (outputRowBytes + 31) & ~31;
    uint32_t inputRowElemsAligned = inputRowBytesAligned / sizeof(float);
    uint32_t outputRowElemsAligned = outputRowBytesAligned / sizeof(float);
    
    pipe.InitBuffer(inBuf, inputRowBytesAligned);
    pipe.InitBuffer(outBuf, outputRowBytesAligned);
    
    for (uint32_t outRow = blockIdx; outRow < totalOutputRows; outRow += blockNum) {
        // Determine which batch/channel/height this output row belongs to
        uint32_t bc = outRow / outputHeight;
        uint32_t oh = outRow % outputHeight;
        uint32_t ih = oh / scaleFactor;
        
        // Input offset for this row
        uint32_t inputOffset = bc * inputHeight * inputWidth + ih * inputWidth;
        // Output offset for this row
        uint32_t outputOffset = bc * outputHeight * outputWidth + oh * outputWidth;
        
        AscendC::LocalTensor<float> inLocal = inBuf.Get<float>();
        AscendC::LocalTensor<float> outLocal = outBuf.Get<float>();
        
        // Copy input row to local
        AscendC::DataCopy(inLocal, xGm[inputOffset], inputRowElemsAligned);
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Nearest neighbor: replicate each pixel scaleFactor times
        for (uint32_t iw = 0; iw < inputWidth; iw++) {
            float val = inLocal.GetValue(iw);
            for (uint32_t s = 0; s < scaleFactor; s++) {
                outLocal.SetValue(iw * scaleFactor + s, val);
            }
        }
        
        AscendC::PipeBarrier<PIPE_ALL>();
        
        // Copy output row to global
        AscendC::DataCopy(yGm[outputOffset], outLocal, outputRowElemsAligned);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
