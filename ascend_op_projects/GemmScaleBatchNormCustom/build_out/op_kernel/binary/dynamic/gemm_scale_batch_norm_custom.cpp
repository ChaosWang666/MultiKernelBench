
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmScaleBatchNorm {
public:
    __aicore__ inline KernelGemmScaleBatchNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR z,
                                uint32_t batchSize, uint32_t featureSize)
    {
        this->featureSize = featureSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerBlock = (batchSize + blockNum - 1) / blockNum;
        uint32_t rowStart = blockIdx * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > batchSize) {
            rowEnd = batchSize;
        }
        this->myRows = (rowEnd > rowStart) ? (rowEnd - rowStart) : 0;

        xGm.SetGlobalBuffer((__gm__ float *)x + (uint64_t)rowStart * featureSize,
                            (uint64_t)this->myRows * featureSize);
        zGm.SetGlobalBuffer((__gm__ float *)z + (uint64_t)rowStart * featureSize,
                            (uint64_t)this->myRows * featureSize);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, featureSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, featureSize * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, featureSize * sizeof(float));
        pipe.InitBuffer(scaleQueue, 1, featureSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0) {
            return;
        }

        AscendC::LocalTensor<float> scaleAlloc = scaleQueue.AllocTensor<float>();
        AscendC::DataCopy(scaleAlloc, scaleGm, this->featureSize);
        scaleQueue.EnQue(scaleAlloc);
        AscendC::LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();

        for (uint32_t i = 0; i < this->myRows; i++) {
            CopyIn(i);
            Compute(i, scaleLocal);
            CopyOut(i);
        }

        scaleQueue.FreeTensor(scaleLocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[(uint64_t)rowIdx * this->featureSize], this->featureSize);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t rowIdx, AscendC::LocalTensor<float>& scaleLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Mul(zLocal, xLocal, scaleLocal, this->featureSize);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[(uint64_t)rowIdx * this->featureSize], zLocal, this->featureSize);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scaleQueue;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    AscendC::GlobalTensor<float> scaleGm;
    uint32_t featureSize;
    uint32_t myRows;
};

extern "C" __global__ __aicore__ void gemm_scale_batch_norm_custom(
    GM_ADDR x, GM_ADDR scale, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmScaleBatchNorm op;
    op.Init(x, scale, z, tiling_data.batchSize, tiling_data.featureSize);
    op.Process();
}
