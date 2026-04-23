
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t C_DIM = 16;
constexpr int32_t C_REP_STRIDE = 2;

class KernelOp {
public:
    __aicore__ inline KernelOp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sub, GM_ADDR y,
                                 uint32_t totalRows, uint32_t rowsPerCore,
                                 uint32_t tileRows_)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startRow_ = blockIdx * rowsPerCore;
        uint32_t endRow = startRow_ + rowsPerCore;
        if (endRow > totalRows) endRow = totalRows;

        this->myStartRow = startRow_;
        this->myRows = (endRow > startRow_) ? (endRow - startRow_) : 0;
        this->tileRows = tileRows_;

        if (this->myRows == 0) return;

        xGm.SetGlobalBuffer((__gm__ float*)x + startRow_ * C_DIM, this->myRows * C_DIM);
        yGm.SetGlobalBuffer((__gm__ float*)y + startRow_, this->myRows);
        subGm.SetGlobalBuffer((__gm__ float*)sub, C_DIM);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileRows_ * C_DIM * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 1024);
        pipe.InitBuffer(subBuf, 64);
        pipe.InitBuffer(tmpBuf1, tileRows_ * C_DIM * sizeof(float));
        pipe.InitBuffer(tmpBuf2, tileRows_ * C_DIM * sizeof(float));
        pipe.InitBuffer(reduceBuf, 1024);
        pipe.InitBuffer(scalarBuf, 64);
    }

    __aicore__ inline void Process()
    {
        if (this->myRows == 0) return;

        AscendC::LocalTensor<float> subLocal = subBuf.Get<float>();
        AscendC::DataCopyExtParams subCopyParams;
        subCopyParams.blockCount = 1;
        subCopyParams.blockLen = C_DIM * sizeof(float);
        subCopyParams.srcStride = 0;
        subCopyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> subPadParams;
        subPadParams.isPad = false;
        subPadParams.leftPadding = 0;
        subPadParams.rightPadding = 0;
        subPadParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(subLocal, subGm, subCopyParams, subPadParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        uint32_t tileNum = (this->myRows + this->tileRows - 1) / this->tileRows;
        for (uint32_t i = 0; i < tileNum; i++) {
            uint32_t curTileRows = this->tileRows;
            if ((i + 1) * this->tileRows > this->myRows) {
                curTileRows = this->myRows - i * this->tileRows;
            }
            CopyIn(i, curTileRows);
            Compute(i, curTileRows, subLocal);
            CopyOut(i, curTileRows);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curRows)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(curRows * C_DIM * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(xLocal, xGm[progress * this->tileRows * C_DIM], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress, uint32_t curRows, AscendC::LocalTensor<float>& subLocal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarTmp = scalarBuf.Get<float>();

        uint32_t totalElems = curRows * C_DIM;

        for (uint32_t row = 0; row < curRows; row++) {
            uint32_t rowOffset = row * C_DIM;

            AscendC::ReduceMax<float>(scalarTmp, xLocal[rowOffset], reduceTmp, C_DIM, false);
            float maxVal = scalarTmp.GetValue(0);

            AscendC::Adds<float>(tmp1[rowOffset], xLocal[rowOffset], -maxVal, C_DIM);
            AscendC::Exp<float>(tmp1[rowOffset], tmp1[rowOffset], C_DIM);

            AscendC::ReduceSum<float, true>(scalarTmp, tmp1[rowOffset], reduceTmp, C_DIM);
            float sumVal = scalarTmp.GetValue(0);
            float invSum = 1.0f / sumVal;

            AscendC::Muls<float>(tmp1[rowOffset], tmp1[rowOffset], invSum, C_DIM);
        }

        AscendC::BinaryRepeatParams binParams;
        binParams.dstBlkStride = 1;
        binParams.src0BlkStride = 1;
        binParams.src1BlkStride = 1;
        binParams.dstRepStride = C_REP_STRIDE;
        binParams.src0RepStride = C_REP_STRIDE;
        binParams.src1RepStride = 0;

        AscendC::Sub<float>(tmp1, tmp1, subLocal, (uint64_t)C_DIM, (uint8_t)curRows, binParams);

        AscendC::Sigmoid<float>(tmp2, tmp1, totalElems);
        AscendC::Mul<float>(tmp1, tmp1, tmp2, totalElems);

        for (uint32_t row = 0; row < curRows; row++) {
            uint32_t rowOffset = row * C_DIM;
            AscendC::ReduceMax<float>(scalarTmp, tmp1[rowOffset], reduceTmp, C_DIM, false);
            yLocal.SetValue(row, scalarTmp.GetValue(0));
        }

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curRows)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(curRows * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[progress * this->tileRows], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> subBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> subGm;
    uint32_t myStartRow;
    uint32_t myRows;
    uint32_t tileRows;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_pool_softmax_subtract_swish_max_custom(GM_ADDR x, GM_ADDR sub, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelOp op;
    op.Init(x, sub, y, tiling_data.totalRows, tiling_data.rowsPerCore, tiling_data.tileRows);
    op.Process();
}
