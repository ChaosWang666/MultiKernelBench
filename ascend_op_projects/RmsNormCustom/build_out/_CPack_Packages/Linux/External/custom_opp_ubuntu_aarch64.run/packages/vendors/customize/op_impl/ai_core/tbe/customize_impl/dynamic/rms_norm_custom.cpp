
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelRmsNorm {
public:
    __aicore__ inline KernelRmsNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t numFeatures, uint32_t spatialSize, float eps)
    {
        this->batchSize = batchSize;
        this->numFeatures = numFeatures;
        this->spatialSize = spatialSize;
        this->eps = eps;
        
        // Total number of (batch, spatial) positions to process
        this->totalPositions = batchSize * spatialSize;
        
        // Each block processes a portion of the total positions
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        uint32_t posPerBlock = (this->totalPositions + blockNum - 1) / blockNum;
        this->startPos = blockIdx * posPerBlock;
        this->endPos = this->startPos + posPerBlock;
        if (this->endPos > this->totalPositions) {
            this->endPos = this->totalPositions;
        }
        
        // Align numFeatures to 32 bytes (8 floats)
        this->alignedFeatures = ((numFeatures + 7) / 8) * 8;
        
        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * numFeatures * spatialSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * numFeatures * spatialSize);
        
        // We need buffers for one feature vector at a time
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->alignedFeatures * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->alignedFeatures * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->alignedFeatures * sizeof(float));
        pipe.InitBuffer(tmpBuf2, BUFFER_NUM, this->alignedFeatures * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t pos = this->startPos; pos < this->endPos; pos++) {
            uint32_t b = pos / this->spatialSize;
            uint32_t s = pos % this->spatialSize;
            ProcessOnePosition(b, s);
        }
    }

private:
    __aicore__ inline void ProcessOnePosition(uint32_t b, uint32_t s)
    {
        // Gather the feature vector x[b, :, s] where layout is (batch, features, spatial)
        // x[b, f, s] is at index b * numFeatures * spatialSize + f * spatialSize + s
        
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> sqLocal = tmpBuf.AllocTensor<float>();
        AscendC::LocalTensor<float> sumLocal = tmpBuf2.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        
        // Gather feature values with stride
        uint32_t baseOffset = b * numFeatures * spatialSize + s;
        for (uint32_t f = 0; f < numFeatures; f++) {
            float val = *(((__gm__ float*)xGm.GetPhyAddr()) + baseOffset + f * spatialSize);
            xLocal.SetValue(f, val);
        }
        // Zero pad aligned portion
        for (uint32_t f = numFeatures; f < alignedFeatures; f++) {
            xLocal.SetValue(f, 0.0f);
        }
        
        // Compute x^2
        AscendC::Mul(sqLocal, xLocal, xLocal, this->alignedFeatures);
        
        // Sum x^2 using ReduceSum
        float sumVal = 0.0f;
        for (uint32_t f = 0; f < numFeatures; f++) {
            sumVal += sqLocal.GetValue(f);
        }
        
        // mean = sum / numFeatures
        float meanVal = sumVal / (float)numFeatures;
        
        // rms = sqrt(mean + eps)
        float rmsVal = 1.0f / sqrtf(meanVal + eps);
        
        // Multiply x by (1/rms)
        AscendC::Muls(yLocal, xLocal, rmsVal, this->alignedFeatures);
        
        // Scatter back to output
        for (uint32_t f = 0; f < numFeatures; f++) {
            *(((__gm__ float*)yGm.GetPhyAddr()) + baseOffset + f * spatialSize) = yLocal.GetValue(f);
        }
        
        tmpBuf2.FreeTensor(sumLocal);
        tmpBuf.FreeTensor(sqLocal);
        outQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpBuf;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t spatialSize;
    uint32_t totalPositions;
    uint32_t startPos;
    uint32_t endPos;
    uint32_t alignedFeatures;
    float eps;
};

extern "C" __global__ __aicore__ void rms_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelRmsNorm op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.numFeatures, tiling_data.spatialSize, tiling_data.eps);
    op.Process();
}
