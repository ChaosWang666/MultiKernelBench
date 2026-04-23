
#include "kernel_operator.h"

class KernelGemmGroupNormMinBiasAdd {
public:
    __aicore__ inline KernelGemmGroupNormMinBiasAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z,
                                uint32_t totalRows, uint32_t totalCols)
    {
        this->totalRows = totalRows;
        this->totalCols = totalCols;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t baseColsPerBlock = totalCols / blockNum;
        uint32_t remainder = totalCols % blockNum;
        if (blockIdx < remainder) {
            this->colsPerBlock = baseColsPerBlock + 1;
            this->colStart = blockIdx * this->colsPerBlock;
        } else {
            this->colsPerBlock = baseColsPerBlock;
            this->colStart = blockIdx * baseColsPerBlock + remainder;
        }

        if (this->colsPerBlock == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, totalRows);
        biasGm.SetGlobalBuffer((__gm__ float*)bias + this->colStart, this->colsPerBlock);
        zGm.SetGlobalBuffer((__gm__ float*)z + this->colStart * totalRows,
                            this->colsPerBlock * totalRows);

        uint32_t rowsAlign = ((totalRows * sizeof(float) + 31) / 32) * 32 / sizeof(float);
        uint32_t biasAlign = ((this->colsPerBlock * sizeof(float) + 31) / 32) * 32 / sizeof(float);

        pipe.InitBuffer(inQueueX, 1, rowsAlign * sizeof(float));
        pipe.InitBuffer(inQueueBias, 1, biasAlign * sizeof(float));
        pipe.InitBuffer(outQueueZ, 2, rowsAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->colsPerBlock == 0) {
            return;
        }

        AscendC::LocalTensor<float> xAlloc = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams xCopyParams;
        xCopyParams.blockCount = 1;
        xCopyParams.blockLen = this->totalRows * sizeof(float);
        xCopyParams.srcStride = 0;
        xCopyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> xPadParams;
        xPadParams.isPad = false;
        xPadParams.leftPadding = 0;
        xPadParams.rightPadding = 0;
        xPadParams.paddingValue = 0;
        AscendC::DataCopyPad(xAlloc, xGm, xCopyParams, xPadParams);
        inQueueX.EnQue(xAlloc);
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();

        AscendC::LocalTensor<float> biasAlloc = inQueueBias.AllocTensor<float>();
        AscendC::DataCopyExtParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = this->colsPerBlock * sizeof(float);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> biasPadParams;
        biasPadParams.isPad = false;
        biasPadParams.leftPadding = 0;
        biasPadParams.rightPadding = 0;
        biasPadParams.paddingValue = 0;
        AscendC::DataCopyPad(biasAlloc, biasGm, biasCopyParams, biasPadParams);
        inQueueBias.EnQue(biasAlloc);
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();

        for (uint32_t j = 0; j < this->colsPerBlock; j++) {
            float biasVal = biasLocal.GetValue(j);
            AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
            AscendC::Adds<float>(outLocal, xLocal, biasVal, this->totalRows);
            outQueueZ.EnQue(outLocal);

            AscendC::LocalTensor<float> outDeq = outQueueZ.DeQue<float>();
            AscendC::DataCopyExtParams outCopyParams;
            outCopyParams.blockCount = 1;
            outCopyParams.blockLen = this->totalRows * sizeof(float);
            outCopyParams.srcStride = 0;
            outCopyParams.dstStride = 0;
            AscendC::DataCopyPad(zGm[j * this->totalRows], outDeq, outCopyParams);
            outQueueZ.FreeTensor(outDeq);
        }

        inQueueX.FreeTensor(xLocal);
        inQueueBias.FreeTensor(biasLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalRows;
    uint32_t totalCols;
    uint32_t colsPerBlock;
    uint32_t colStart;
};

extern "C" __global__ __aicore__ void gemm_group_norm_min_bias_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmGroupNormMinBiasAdd op;
    op.Init(x, bias, z, tiling_data.totalRows, tiling_data.totalCols);
    op.Process();
}
