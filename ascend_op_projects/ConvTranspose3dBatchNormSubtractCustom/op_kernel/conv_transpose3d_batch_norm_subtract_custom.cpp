
#include "kernel_operator.h"

class KernelMeanSub {
public:
    __aicore__ inline KernelMeanSub() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalRows, uint32_t rowLength, uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();
        uint32_t rowsPerBlock = (totalRows + numBlocks - 1) / numBlocks;
        uint32_t startRow = blockIdx * rowsPerBlock;
        uint32_t endRow = startRow + rowsPerBlock;
        if (endRow > totalRows) { endRow = totalRows; }
        if (startRow > totalRows) { startRow = totalRows; }

        this->startRow = startRow;
        this->rowsThisBlock = endRow - startRow;
        this->rowLength = rowLength;
        this->tileLength = tileLength;
        this->invRowLength = 1.0f / (float)rowLength;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)totalRows * rowLength);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)totalRows * rowLength);

        pipe.InitBuffer(inQueueX, 2, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 2, tileLength * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, 16 * 1024);
        pipe.InitBuffer(reduceDstBuf, 64);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < rowsThisBlock; r++) {
            uint64_t rowOffset = (uint64_t)(startRow + r) * (uint64_t)rowLength;
            float sum = ComputeSum(rowOffset);
            float mean = sum * invRowLength;
            SubtractMean(rowOffset, mean);
        }
    }

private:
    __aicore__ inline float ComputeSum(uint64_t rowOffset)
    {
        float totalSum = 0.0f;
        uint32_t numTiles = (rowLength + tileLength - 1) / tileLength;

        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> reduceDst = reduceDstBuf.Get<float>();

        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t offset = t * tileLength;
            uint32_t len = (offset + tileLength <= rowLength) ? tileLength : (rowLength - offset);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = len * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0.0f;

            AscendC::DataCopyPad(xLocal, xGm[rowOffset + offset], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::ReduceSum<float, true>(reduceDst, xIn, reduceTmp, (int32_t)len);

            AscendC::PipeBarrier<PIPE_V>();
            totalSum += reduceDst.GetValue(0);

            inQueueX.FreeTensor(xIn);
        }

        return totalSum;
    }

    __aicore__ inline void SubtractMean(uint64_t rowOffset, float mean)
    {
        uint32_t numTiles = (rowLength + tileLength - 1) / tileLength;
        float negMean = -mean;

        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t offset = t * tileLength;
            uint32_t len = (offset + tileLength <= rowLength) ? tileLength : (rowLength - offset);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0.0f;
            AscendC::DataCopyPad(xLocal, xGm[rowOffset + offset], copyInParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            AscendC::Adds<float>(yLocal, xIn, negMean, len);
            outQueueY.EnQue<float>(yLocal);
            inQueueX.FreeTensor(xIn);

            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = len * sizeof(float);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            AscendC::DataCopyPad(yGm[rowOffset + offset], yOut, copyOutParams);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceDstBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t startRow;
    uint32_t rowsThisBlock;
    uint32_t rowLength;
    uint32_t tileLength;
    float invRowLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_batch_norm_subtract_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMeanSub op;
    op.Init(x, y, tiling_data.totalRows, tiling_data.rowLength, tiling_data.tileLength);
    op.Process();
}
