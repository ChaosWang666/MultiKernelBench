
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelSoftmaxSigmoid {
public:
    __aicore__ inline KernelSoftmaxSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t channels, uint32_t spatialSize, uint32_t batchSize, uint32_t tileNum)
    {
        this->channels = channels;
        this->spatialSize = spatialSize;
        this->batchSize = batchSize;
        
        // Total spatial positions across all batches
        uint32_t totalSpatial = batchSize * spatialSize;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Divide spatial positions across blocks
        uint32_t spatialPerBlock = (totalSpatial + blockNum - 1) / blockNum;
        this->startSpatial = blockIdx * spatialPerBlock;
        this->endSpatial = (this->startSpatial + spatialPerBlock > totalSpatial) ? totalSpatial : this->startSpatial + spatialPerBlock;
        if (this->startSpatial >= totalSpatial) {
            this->startSpatial = totalSpatial;
            this->endSpatial = totalSpatial;
        }
        this->mySpatialCount = this->endSpatial - this->startSpatial;
        
        this->tileNum = tileNum;
        if (this->mySpatialCount < tileNum) {
            this->tileNum = (this->mySpatialCount > 0) ? this->mySpatialCount : 1;
        }
        
        // Align tile length to 32 bytes (8 floats)
        uint32_t alignUnit = 8;
        this->tileLength = this->mySpatialCount / this->tileNum;
        // Make sure tileLength is aligned
        this->tileLength = (this->tileLength / alignUnit) * alignUnit;
        if (this->tileLength == 0) this->tileLength = alignUnit;
        
        // Recalculate tileNum based on aligned tileLength
        if (this->mySpatialCount > 0) {
            this->tileNum = this->mySpatialCount / this->tileLength;
            this->tailLength = this->mySpatialCount - this->tileNum * this->tileLength;
            // align tail
            this->tailLengthAligned = ((this->tailLength + alignUnit - 1) / alignUnit) * alignUnit;
        } else {
            this->tileNum = 0;
            this->tailLength = 0;
            this->tailLengthAligned = 0;
        }
        
        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);
        
        // We need buffers for channels elements per spatial position
        // For tileLength spatial positions, we need channels * tileLength floats
        // But we process one channel-vector at a time
        // Strategy: for each tile of spatial positions, load all channels, compute softmax+sigmoid, store
        
        // Buffer for one channel slice of tileLength spatial positions
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        // Temp buffers for max, sum, exp values
        pipe.InitBuffer(maxBuf, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(sumBuf, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->tileLength * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        // Process each tile of spatial positions
        for (uint32_t t = 0; t < this->tileNum; t++) {
            ProcessTile(t, this->tileLength);
        }
        // Process tail
        if (this->tailLength > 0) {
            ProcessTail();
        }
    }

private:
    __aicore__ inline void GetGlobalIndex(uint32_t spatialIdx, uint32_t channel, uint32_t &globalIdx)
    {
        // Layout is (batch, channels, D, H, W) = (batch, channels, spatialSize)
        // spatialIdx is a flattened index across all batches' spatial positions
        uint32_t batchIdx = spatialIdx / this->spatialSize;
        uint32_t localSpatial = spatialIdx - batchIdx * this->spatialSize;
        globalIdx = batchIdx * (this->channels * this->spatialSize) + channel * this->spatialSize + localSpatial;
    }
    
    __aicore__ inline void ProcessTile(uint32_t tileIdx, uint32_t len)
    {
        uint32_t spatialStart = this->startSpatial + tileIdx * this->tileLength;
        
        // Step 1: Find max across channels for each spatial position
        AscendC::LocalTensor<float> maxLocal = maxBuf.Get<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        
        // Load channel 0 as initial max
        for (uint32_t s = 0; s < len; s++) {
            uint32_t gIdx;
            GetGlobalIndex(spatialStart + s, 0, gIdx);
            maxLocal.SetValue(s, xGm.GetValue(gIdx));
        }
        
        // Find max across all channels
        for (uint32_t c = 1; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float curMax = maxLocal.GetValue(s);
                if (val > curMax) {
                    maxLocal.SetValue(s, val);
                }
            }
        }
        
        // Step 2: Compute sum of exp(x - max) across channels
        for (uint32_t s = 0; s < len; s++) {
            sumLocal.SetValue(s, 0.0f);
        }
        
        for (uint32_t c = 0; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float maxVal = maxLocal.GetValue(s);
                float expVal = exp(val - maxVal);
                float curSum = sumLocal.GetValue(s);
                sumLocal.SetValue(s, curSum + expVal);
            }
        }
        
        // Step 3: Compute softmax then sigmoid for each channel and store
        for (uint32_t c = 0; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float maxVal = maxLocal.GetValue(s);
                float sumVal = sumLocal.GetValue(s);
                float softmaxVal = exp(val - maxVal) / sumVal;
                float sigmoidVal = 1.0f / (1.0f + exp(-softmaxVal));
                yGm.SetValue(gIdx, sigmoidVal);
            }
        }
    }
    
    __aicore__ inline void ProcessTail()
    {
        uint32_t spatialStart = this->startSpatial + this->tileNum * this->tileLength;
        uint32_t len = this->tailLength;
        
        AscendC::LocalTensor<float> maxLocal = maxBuf.Get<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        
        for (uint32_t s = 0; s < len; s++) {
            uint32_t gIdx;
            GetGlobalIndex(spatialStart + s, 0, gIdx);
            maxLocal.SetValue(s, xGm.GetValue(gIdx));
        }
        
        for (uint32_t c = 1; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float curMax = maxLocal.GetValue(s);
                if (val > curMax) {
                    maxLocal.SetValue(s, val);
                }
            }
        }
        
        for (uint32_t s = 0; s < len; s++) {
            sumLocal.SetValue(s, 0.0f);
        }
        
        for (uint32_t c = 0; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float maxVal = maxLocal.GetValue(s);
                float expVal = exp(val - maxVal);
                float curSum = sumLocal.GetValue(s);
                sumLocal.SetValue(s, curSum + expVal);
            }
        }
        
        for (uint32_t c = 0; c < this->channels; c++) {
            for (uint32_t s = 0; s < len; s++) {
                uint32_t gIdx;
                GetGlobalIndex(spatialStart + s, c, gIdx);
                float val = xGm.GetValue(gIdx);
                float maxVal = maxLocal.GetValue(s);
                float sumVal = sumLocal.GetValue(s);
                float softmaxVal = exp(val - maxVal) / sumVal;
                float sigmoidVal = 1.0f / (1.0f + exp(-softmaxVal));
                yGm.SetValue(gIdx, sigmoidVal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf, sumBuf, tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t channels;
    uint32_t spatialSize;
    uint32_t batchSize;
    uint32_t startSpatial;
    uint32_t endSpatial;
    uint32_t mySpatialCount;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailLength;
    uint32_t tailLengthAligned;
};

extern "C" __global__ __aicore__ void conv_transpose3d_softmax_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmaxSigmoid op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.channels, tiling_data.spatialSize, tiling_data.batchSize, tiling_data.tileNum);
    op.Process();
}
