
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelInstanceNormResidualAddMul {
public:
    __aicore__ inline KernelInstanceNormResidualAddMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                 uint32_t totalRows, uint32_t rowLen)
    {
        this->rowLen = rowLen;
        this->eps = 1e-5f;
        this->invRowLen = 1.0f / (float)rowLen;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();

        uint32_t rowsPerBlock = (totalRows + numBlocks - 1) / numBlocks;
        uint32_t startRow = blockIdx * rowsPerBlock;
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > totalRows) endRow = totalRows;
        if (startRow > totalRows) startRow = totalRows;

        this->startRow = startRow;
        this->endRow = endRow;

        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows * rowLen);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalRows * rowLen);
        zGm.SetGlobalBuffer((__gm__ float*)z, totalRows * rowLen);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, rowLen * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, rowLen * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, rowLen * sizeof(float));
        pipe.InitBuffer(tmpBuf, rowLen * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32 * 1024);
        pipe.InitBuffer(sumBuf, 64);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = this->startRow; row < this->endRow; row++) {
            CopyIn(row);
            Compute(row);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[row * this->rowLen], this->rowLen);
        AscendC::DataCopy(yLocal, yGm[row * this->rowLen], this->rowLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t row)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();

        // 1. sum = ReduceSum(x), mean = sum / rowLen
        AscendC::ReduceSum<float, true>(sumLocal, xLocal, reduceTmp, this->rowLen);
        float sumVal = sumLocal.GetValue(0);
        float mean = sumVal * this->invRowLen;

        // 2. tmpLocal = x - mean
        AscendC::Adds<float>(tmpLocal, xLocal, -mean, this->rowLen);

        // 3. zLocal = tmpLocal * tmpLocal (for variance)
        AscendC::Mul<float>(zLocal, tmpLocal, tmpLocal, this->rowLen);

        // 4. sumSq = ReduceSum(zLocal), var = sumSq / rowLen
        AscendC::ReduceSum<float, true>(sumLocal, zLocal, reduceTmp, this->rowLen);
        float sumSq = sumLocal.GetValue(0);
        float var = sumSq * this->invRowLen;

        // 5. invStd = 1 / sqrt(var + eps) via tensor sqrt
        AscendC::Duplicate<float>(sumLocal, var + this->eps, 8);
        AscendC::Sqrt<float>(sumLocal, sumLocal, 8);
        float stdVal = sumLocal.GetValue(0);
        float invStd = 1.0f / stdVal;

        // 6. tmpLocal = (x - mean) * invStd
        AscendC::Muls<float>(tmpLocal, tmpLocal, invStd, this->rowLen);

        // 7. zLocal = x_norm + y
        AscendC::Add<float>(zLocal, tmpLocal, yLocal, this->rowLen);

        // 8. zLocal = zLocal * y
        AscendC::Mul<float>(zLocal, zLocal, yLocal, this->rowLen);

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t row)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[row * this->rowLen], zLocal, this->rowLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t rowLen;
    uint32_t startRow;
    uint32_t endRow;
    float eps;
    float invRowLen;
};

extern "C" __global__ __aicore__ void bmm_instance_norm_sum_residual_add_multiply_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInstanceNormResidualAddMul op;
    op.Init(x, y, z, tiling_data.totalRows, tiling_data.rowLen);
    op.Process();
}
