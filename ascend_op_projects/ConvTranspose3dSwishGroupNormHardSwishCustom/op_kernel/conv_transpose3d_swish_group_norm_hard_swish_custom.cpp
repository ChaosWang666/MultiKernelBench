
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelHardSwish {
public:
    __aicore__ inline KernelHardSwish() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t baseLen = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;

        uint32_t myStart;
        uint32_t myLen;
        if (blockIdx < remainder) {
            myLen = baseLen + 1;
            myStart = blockIdx * myLen;
        } else {
            myLen = baseLen;
            myStart = remainder * (baseLen + 1) + (blockIdx - remainder) * baseLen;
        }

        this->myLen = myLen;
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + myStart, myLen);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + myStart, myLen);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(constBuf, tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t processed = 0;
        while (processed < myLen) {
            uint32_t remaining = myLen - processed;
            uint32_t curLen = (remaining < tileLength) ? remaining : tileLength;
            CopyIn(processed, curLen);
            Compute(curLen);
            CopyOut(processed, curLen);
            processed += curLen;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (uint32_t)(len * sizeof(DTYPE_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DTYPE_X> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> cst = constBuf.Get<DTYPE_X>();

        AscendC::Adds(tmp, xLocal, (DTYPE_X)3.0f, len);
        AscendC::Duplicate(cst, (DTYPE_X)6.0f, len);
        AscendC::Min(tmp, tmp, cst, len);
        AscendC::Duplicate(cst, (DTYPE_X)0.0f, len);
        AscendC::Max(tmp, tmp, cst, len);
        AscendC::Mul(yLocal, tmp, xLocal, len);
        AscendC::Muls(yLocal, yLocal, (DTYPE_X)(1.0f / 6.0f), len);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (uint32_t)(len * sizeof(DTYPE_Y));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> constBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t myLen;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv_transpose3d_swish_group_norm_hard_swish_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHardSwish op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
