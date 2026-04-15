
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFrobeniusNorm {
public:
    __aicore__ inline KernelFrobeniusNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->blockNum = AscendC::GetBlockNum();
        this->blockIdx = AscendC::GetBlockIdx();
        this->blockLength = totalLength / this->blockNum;
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * this->blockIdx, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * this->blockIdx, this->blockLength);
        workGm.SetGlobalBuffer((__gm__ float *)workspace, this->blockNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(reduceBuf, 1, 32);  // small buffer for reduce result
    }

    __aicore__ inline void Process()
    {
        // Phase 1: compute local sum of squares
        float localSum = 0.0f;
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();
            AscendC::LocalTensor<float> redLocal = reduceBuf.Get<float>();

            // xIn * xIn -> tmp
            AscendC::Mul(tmp, xIn, xIn, this->tileLength);
            // reduce sum
            AscendC::ReduceSum(redLocal, tmp, this->tileLength, false);
            localSum += redLocal.GetValue(0);

            inQueueX.FreeTensor(xIn);
        }

        // Phase 2: write local partial sum to workspace
        workGm.SetValue(this->blockIdx, localSum);

        // Synchronize all blocks
        AscendC::SyncAll();

        // Phase 3: compute global norm from workspace
        float globalSum = 0.0f;
        for (uint32_t i = 0; i < this->blockNum; i++) {
            globalSum += workGm.GetValue(i);
        }
        float normVal = 1.0f;
        if (globalSum > 0.0f) {
            // Use reciprocal of sqrt
            // sqrt approximation
            float sqrtVal = globalSum;
            // Newton's method for sqrt
            for (int iter = 0; iter < 20; iter++) {
                sqrtVal = 0.5f * (sqrtVal + globalSum / sqrtVal);
            }
            normVal = sqrtVal;
        }

        // Phase 4: divide each element by norm
        float invNorm = 1.0f / normVal;
        for (int32_t i = 0; i < loopCount; i++) {
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            AscendC::Muls(yLocal, xIn, invNorm, this->tileLength);

            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xIn);

            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopy(yGm[i * this->tileLength], yOut, this->tileLength);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> workGm;
    uint32_t totalLength;
    uint32_t blockNum;
    uint32_t blockIdx;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void frobenius_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFrobeniusNorm op;
    op.Init(x, y, workspace, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
