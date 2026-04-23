
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulAvgPoolGeluScaleMax {
public:
    __aicore__ inline KernelMatmulAvgPoolGeluScaleMax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchSize, uint32_t outFeatures,
                                 uint32_t poolKernelSize, uint32_t pooledSize,
                                 float scaleFactor, float invPoolSize)
    {
        this->batchSize = batchSize;
        this->outFeatures = outFeatures;
        this->poolKernelSize = poolKernelSize;
        this->pooledSize = pooledSize;
        this->scaleFactor = scaleFactor;
        this->invPoolSize = invPoolSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t totalBlocks = AscendC::GetBlockNum();

        uint32_t rowsPerBlock = batchSize / totalBlocks;
        uint32_t remainder = batchSize % totalBlocks;
        if (blockIdx < remainder) {
            this->startRow = blockIdx * (rowsPerBlock + 1);
            this->rowCount = rowsPerBlock + 1;
        } else {
            this->startRow = blockIdx * rowsPerBlock + remainder;
            this->rowCount = rowsPerBlock;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * outFeatures);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, outFeatures * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
        pipe.InitBuffer(pooledBuf, pooledSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, pooledSize * sizeof(float));
        pipe.InitBuffer(patternTmpBuf, 16384);
        pipe.InitBuffer(reduceTmpBuf, 1024);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < rowCount; r++) {
            uint32_t rowIdx = startRow + r;
            CopyIn(rowIdx);
            Compute();
            CopyOut(rowIdx);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * outFeatures], outFeatures);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> pooledLocal = pooledBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::LocalTensor<uint8_t> patternTmp = patternTmpBuf.Get<uint8_t>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        uint32_t srcShape[2] = {pooledSize, poolKernelSize};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            pooledLocal, xLocal, patternTmp, srcShape, true);

        AscendC::Muls<float>(pooledLocal, pooledLocal, invPoolSize, pooledSize);

        AscendC::Gelu<float>(tmpLocal, pooledLocal, pooledSize);

        AscendC::Muls<float>(tmpLocal, tmpLocal, scaleFactor, pooledSize);

        AscendC::ReduceMax<float>(yLocal, tmpLocal, reduceTmp, (int32_t)pooledSize, false);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(yGm[rowIdx], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pooledBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> patternTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t outFeatures;
    uint32_t poolKernelSize;
    uint32_t pooledSize;
    float scaleFactor;
    float invPoolSize;
    uint32_t startRow;
    uint32_t rowCount;
};

extern "C" __global__ __aicore__ void matmul_avg_pool_gelu_scale_max_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulAvgPoolGeluScaleMax op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.outFeatures,
            tiling_data.poolKernelSize, tiling_data.pooledSize,
            tiling_data.scaleFactor, tiling_data.invPoolSize);
    op.Process();
}
