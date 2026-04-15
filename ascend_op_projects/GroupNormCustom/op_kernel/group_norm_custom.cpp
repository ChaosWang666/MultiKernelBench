
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGroupNorm {
public:
    __aicore__ inline KernelGroupNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                 uint32_t batchSize, uint32_t numGroups, uint32_t numChannels,
                                 uint32_t numHW, uint32_t channelsPerGroup, uint32_t groupSize, float eps)
    {
        this->batchSize = batchSize;
        this->numGroups = numGroups;
        this->numChannels = numChannels;
        this->numHW = numHW;
        this->channelsPerGroup = channelsPerGroup;
        this->groupSize = groupSize;
        this->eps = eps;

        uint32_t totalTasks = batchSize * numGroups;
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->taskStart = blockIdx * ((totalTasks + numBlocks - 1) / numBlocks);
        this->taskEnd = (blockIdx + 1) * ((totalTasks + numBlocks - 1) / numBlocks);
        if (this->taskEnd > totalTasks) this->taskEnd = totalTasks;

        uint32_t totalLength = batchSize * numChannels * numHW;
        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, numChannels);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, numChannels);

        // Align tile length to 32 bytes (8 floats)
        uint32_t alignedHW = ((numHW + 7) / 8) * 8;
        this->alignedHW = alignedHW;

        // We process one channel at a time from the group
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedHW * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, alignedHW * sizeof(float));
        pipe.InitBuffer(tmpBuf1, BUFFER_NUM, alignedHW * sizeof(float));
        pipe.InitBuffer(tmpBuf2, BUFFER_NUM, alignedHW * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t task = this->taskStart; task < this->taskEnd; task++) {
            uint32_t batchIdx = task / this->numGroups;
            uint32_t groupIdx = task % this->numGroups;
            ProcessOneGroup(batchIdx, groupIdx);
        }
    }

private:
    __aicore__ inline void ProcessOneGroup(uint32_t batchIdx, uint32_t groupIdx)
    {
        uint32_t channelStart = groupIdx * this->channelsPerGroup;
        uint32_t batchOffset = batchIdx * this->numChannels * this->numHW;
        
        // Two-pass approach:
        // Pass 1: compute mean and variance
        // Pass 2: normalize, scale, and shift
        
        float sum = 0.0f;
        float sumSq = 0.0f;
        float groupSizeF = (float)this->groupSize;
        
        // Pass 1: compute sum and sum of squares
        for (uint32_t c = 0; c < this->channelsPerGroup; c++) {
            uint32_t channelIdx = channelStart + c;
            uint32_t offset = batchOffset + channelIdx * this->numHW;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[offset], this->alignedHW);
            inQueueX.EnQue(xLocal);
            
            xLocal = inQueueX.DeQue<float>();
            
            // Compute partial sum
            AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
            
            // sum of elements
            float localSum = 0.0f;
            float localSumSq = 0.0f;
            
            AscendC::ReduceSum(tmp1, xLocal, tmpBuf2, this->alignedHW);
            localSum = tmp1.GetValue(0);
            
            // sum of squares
            AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();
            AscendC::Mul(tmp2, xLocal, xLocal, this->alignedHW);
            tmpBuf2.FreeTensor(tmp2);
            
            tmp2 = tmpBuf2.AllocTensor<float>();
            // Need another temp buffer for ReduceSum; reuse carefully
            AscendC::LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
            AscendC::Mul(outLocal, xLocal, xLocal, this->alignedHW);
            AscendC::ReduceSum(tmp2, outLocal, tmp1, this->alignedHW);
            localSumSq = tmp2.GetValue(0);
            outQueueY.FreeTensor(outLocal);
            tmpBuf2.FreeTensor(tmp2);
            
            // Adjust for aligned vs actual HW
            // If alignedHW > numHW, the extra elements from DataCopy might be garbage
            // We need to handle that - but for simplicity assume numHW is aligned or handle tail
            // Actually we must zero out extra or subtract extra contributions
            // For correctness, let's handle the tail if numHW != alignedHW
            if (this->alignedHW > this->numHW) {
                for (uint32_t i = this->numHW; i < this->alignedHW; i++) {
                    float v = xLocal.GetValue(i);
                    localSum -= v;
                    localSumSq -= v * v;
                }
            }
            
            sum += localSum;
            sumSq += localSumSq;
            
            tmpBuf1.FreeTensor(tmp1);
            inQueueX.FreeTensor(xLocal);
        }
        
        float mean = sum / groupSizeF;
        float var = sumSq / groupSizeF - mean * mean;
        float invStd = 1.0f / sqrtf(var + this->eps);
        
        // Pass 2: normalize
        for (uint32_t c = 0; c < this->channelsPerGroup; c++) {
            uint32_t channelIdx = channelStart + c;
            uint32_t offset = batchOffset + channelIdx * this->numHW;
            
            float gamma_val = gammaGm.GetValue(channelIdx);
            float beta_val = betaGm.GetValue(channelIdx);
            float scale = gamma_val * invStd;
            float bias = beta_val - mean * scale;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[offset], this->alignedHW);
            inQueueX.EnQue(xLocal);
            
            xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
            
            // outLocal = xLocal * scale + bias
            AscendC::Muls(outLocal, xLocal, scale, this->alignedHW);
            AscendC::Adds(outLocal, outLocal, bias, this->alignedHW);
            
            outQueueY.EnQue(outLocal);
            outLocal = outQueueY.DeQue<float>();
            
            AscendC::DataCopy(yGm[offset], outLocal, this->alignedHW);
            
            outQueueY.FreeTensor(outLocal);
            inQueueX.FreeTensor(xLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::GlobalTensor<float> xGm, yGm, gammaGm, betaGm;
    uint32_t batchSize, numGroups, numChannels, numHW, channelsPerGroup, groupSize;
    uint32_t alignedHW;
    float eps;
    uint32_t taskStart, taskEnd;
};

extern "C" __global__ __aicore__ void group_norm_custom(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupNorm op;
    op.Init(x, gamma, beta, y,
            tiling_data.batchSize, tiling_data.numGroups, tiling_data.numChannels,
            tiling_data.numHW, tiling_data.channelsPerGroup, tiling_data.groupSize, tiling_data.eps);
    op.Process();
}
