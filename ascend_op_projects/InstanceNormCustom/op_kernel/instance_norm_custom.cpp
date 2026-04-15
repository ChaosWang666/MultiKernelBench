
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelInstanceNorm {
public:
    __aicore__ inline KernelInstanceNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t numFeatures, uint32_t spatialSize, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->numFeatures = numFeatures;
        this->spatialSize = spatialSize;
        this->tileNum = tileNum;

        // Total number of (batch, channel) instances
        this->totalInstances = batchSize * numFeatures;
        
        // Each block processes a subset of instances
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->instancesPerBlock = (this->totalInstances + blockNum - 1) / blockNum;
        this->startInstance = blockIdx * this->instancesPerBlock;
        this->endInstance = this->startInstance + this->instancesPerBlock;
        if (this->endInstance > this->totalInstances) {
            this->endInstance = this->totalInstances;
        }

        // Align tileLength to 32 bytes (8 floats)
        uint32_t rawTileLength = spatialSize / tileNum;
        this->tileLength = (rawTileLength + 7) & ~7;  // align up to 8
        if (this->tileLength > spatialSize) {
            this->tileLength = (spatialSize + 7) & ~7;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * numFeatures * spatialSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * numFeatures * spatialSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));
        // Work buffer for reduce sum - needs to be large enough
        pipe.InitBuffer(workBuf, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t inst = this->startInstance; inst < this->endInstance; inst++) {
            ProcessOneInstance(inst);
        }
    }

private:
    __aicore__ inline void ProcessOneInstance(uint32_t instIdx)
    {
        uint32_t baseOffset = instIdx * this->spatialSize;
        float eps = 1e-5f;

        // First pass: compute mean
        float sumTotal = 0.0f;
        uint32_t remaining = this->spatialSize;
        uint32_t offset = 0;
        
        while (remaining > 0) {
            uint32_t curLen = remaining > this->tileLength ? this->tileLength : remaining;
            // Align curLen up to 8
            uint32_t alignedLen = (curLen + 7) & ~7;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            // Zero out the buffer first if alignedLen > curLen
            if (alignedLen > curLen) {
                AscendC::Duplicate(xLocal, 0.0f, alignedLen);
            }
            AscendC::DataCopy(xLocal, xGm[baseOffset + offset], alignedLen);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xProc = inQueueX.DeQue<float>();
            // Zero out padding elements
            if (curLen < alignedLen) {
                for (uint32_t i = curLen; i < alignedLen; i++) {
                    xProc.SetValue(i, 0.0f);
                }
            }
            
            AscendC::LocalTensor<float> workLocal = workBuf.Get<float>();
            float partialSum = 0.0f;
            // Sum reduction
            AscendC::LocalTensor<float> tmpLocal = tmpBuf1.Get<float>();
            AscendC::ReduceSum(tmpLocal, xProc, workLocal, alignedLen);
            partialSum = tmpLocal.GetValue(0);
            sumTotal += partialSum;

            inQueueX.FreeTensor(xProc);
            offset += curLen;
            remaining -= curLen;
        }

        float mean = sumTotal / (float)this->spatialSize;

        // Second pass: compute variance
        float varTotal = 0.0f;
        remaining = this->spatialSize;
        offset = 0;

        while (remaining > 0) {
            uint32_t curLen = remaining > this->tileLength ? this->tileLength : remaining;
            uint32_t alignedLen = (curLen + 7) & ~7;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            if (alignedLen > curLen) {
                AscendC::Duplicate(xLocal, 0.0f, alignedLen);
            }
            AscendC::DataCopy(xLocal, xGm[baseOffset + offset], alignedLen);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xProc = inQueueX.DeQue<float>();
            if (curLen < alignedLen) {
                for (uint32_t i = curLen; i < alignedLen; i++) {
                    xProc.SetValue(i, 0.0f);
                }
            }

            // x - mean  
            AscendC::LocalTensor<float> tmpLocal = tmpBuf1.Get<float>();
            AscendC::Adds(tmpLocal, xProc, -mean, alignedLen);
            // (x - mean)^2
            AscendC::LocalTensor<float> tmpLocal2 = tmpBuf2.Get<float>();
            AscendC::Mul(tmpLocal2, tmpLocal, tmpLocal, alignedLen);
            // Zero padding for variance
            if (curLen < alignedLen) {
                for (uint32_t i = curLen; i < alignedLen; i++) {
                    tmpLocal2.SetValue(i, 0.0f);
                }
            }
            
            AscendC::LocalTensor<float> workLocal = workBuf.Get<float>();
            AscendC::LocalTensor<float> sumOut = tmpBuf1.Get<float>();
            AscendC::ReduceSum(sumOut, tmpLocal2, workLocal, alignedLen);
            varTotal += sumOut.GetValue(0);

            inQueueX.FreeTensor(xProc);
            offset += curLen;
            remaining -= curLen;
        }

        float variance = varTotal / (float)this->spatialSize;
        float invStd = 1.0f / sqrtf(variance + eps);

        // Third pass: normalize and write output
        remaining = this->spatialSize;
        offset = 0;

        while (remaining > 0) {
            uint32_t curLen = remaining > this->tileLength ? this->tileLength : remaining;
            uint32_t alignedLen = (curLen + 7) & ~7;
            
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            if (alignedLen > curLen) {
                AscendC::Duplicate(xLocal, 0.0f, alignedLen);
            }
            AscendC::DataCopy(xLocal, xGm[baseOffset + offset], alignedLen);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xProc = inQueueX.DeQue<float>();
            
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            // (x - mean) * invStd
            AscendC::Adds(yLocal, xProc, -mean, alignedLen);
            AscendC::Muls(yLocal, yLocal, invStd, alignedLen);

            outQueueY.EnQue(yLocal);
            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopy(yGm[baseOffset + offset], yOut, alignedLen);
            outQueueY.FreeTensor(yOut);

            inQueueX.FreeTensor(xProc);
            offset += curLen;
            remaining -= curLen;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2, workBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t numFeatures;
    uint32_t spatialSize;
    uint32_t tileNum;
    uint32_t totalInstances;
    uint32_t instancesPerBlock;
    uint32_t startInstance;
    uint32_t endInstance;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void instance_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInstanceNorm op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.numFeatures, tiling_data.spatialSize, tiling_data.tileNum);
    op.Process();
}
