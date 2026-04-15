
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMinTanhTanh {
public:
    __aicore__ inline KernelMinTanhTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum, uint32_t channels, uint32_t spatialSize)
    {
        this->channels = channels;
        this->spatialSize = spatialSize;
        // totalLength = batch * spatialSize (number of output elements)
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        
        // Align tileLength to 32 bytes (8 floats)
        if (this->tileLength < 8) {
            this->tileLength = 8;
        }
        this->tileLength = (this->tileLength / 8) * 8;
        
        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();
        // Compute which batch and spatial offset this block starts at
        // x is [batch, channels, H, W], y is [batch, 1, H, W] flattened
        // For output element i (in block-local), global output index = blockOffset + i
        // global output index maps to batch = idx / spatialSize, spatial = idx % spatialSize
        // input for that element: x[batch, c, spatial] for all c
        
        this->globalOutOffset = blockOffset;
        
        yGm.SetGlobalBuffer((__gm__ float *)y + blockOffset, this->blockLength);
        xGm.SetGlobalBuffer((__gm__ float *)x, channels * spatialSize * ((totalLength + spatialSize - 1) / spatialSize));
        
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->tileLength * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            ComputeTile(i);
        }
    }

private:
    __aicore__ inline void ComputeTile(int32_t progress)
    {
        uint32_t offset = progress * this->tileLength;
        uint32_t len = this->tileLength;
        
        AscendC::LocalTensor<float> minLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> chanLocal = inQueueX.AllocTensor<float>();
        
        // Load first channel data as initial min
        // For each element j in [0, len), global output index = globalOutOffset + offset + j
        // batch = globalIdx / spatialSize, sp = globalIdx % spatialSize
        // x index = batch * channels * spatialSize + c * spatialSize + sp
        // We need to gather from x for channel 0 first
        for (uint32_t j = 0; j < len; j++) {
            uint32_t globalIdx = this->globalOutOffset + offset + j;
            uint32_t batch = globalIdx / this->spatialSize;
            uint32_t sp = globalIdx % this->spatialSize;
            uint32_t xIdx = batch * this->channels * this->spatialSize + sp;
            minLocal.SetValue(j, xGm.GetValue(xIdx));
        }
        
        // Iterate over remaining channels, take element-wise min
        for (uint32_t c = 1; c < this->channels; c++) {
            for (uint32_t j = 0; j < len; j++) {
                uint32_t globalIdx = this->globalOutOffset + offset + j;
                uint32_t batch = globalIdx / this->spatialSize;
                uint32_t sp = globalIdx % this->spatialSize;
                uint32_t xIdx = batch * this->channels * this->spatialSize + c * this->spatialSize + sp;
                float val = xGm.GetValue(xIdx);
                float curMin = minLocal.GetValue(j);
                if (val < curMin) {
                    minLocal.SetValue(j, val);
                }
            }
        }
        
        // Apply tanh twice using vector operations
        // We need minLocal in a proper queue for vector ops
        // Copy result to chanLocal, do tanh, copy back
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        
        // First tanh
        for (uint32_t j = 0; j < len; j++) {
            chanLocal.SetValue(j, minLocal.GetValue(j));
        }
        
        // Compute tanh element-wise (scalar fallback)
        for (uint32_t j = 0; j < len; j++) {
            float v = chanLocal.GetValue(j);
            // tanh approximation using exp
            float ep = exp(2.0f * v);
            float t = (ep - 1.0f) / (ep + 1.0f);
            chanLocal.SetValue(j, t);
        }
        
        // Second tanh
        for (uint32_t j = 0; j < len; j++) {
            float v = chanLocal.GetValue(j);
            float ep = exp(2.0f * v);
            float t = (ep - 1.0f) / (ep + 1.0f);
            minLocal.SetValue(j, t);
        }
        
        // Copy out
        outQueueZ.EnQue<float>(minLocal);
        inQueueX.FreeTensor(chanLocal);
        
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(yGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t channels;
    uint32_t spatialSize;
    uint32_t globalOutOffset;
};

extern "C" __global__ __aicore__ void conv2d_min_tanh_tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMinTanhTanh op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum, tiling_data.channels, tiling_data.spatialSize);
    op.Process();
}
