
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelClamp {
public:
    __aicore__ inline KernelClamp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileLength)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t basePerBlock = totalLength / blockNum;
        uint32_t remainder = totalLength % blockNum;
        this->myLength = basePerBlock + (blockIdx < remainder ? 1 : 0);
        uint32_t blockStart = blockIdx * basePerBlock + (blockIdx < remainder ? blockIdx : remainder);
        this->tileLength = tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockStart, this->myLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + blockStart, this->myLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        uint32_t offset = 0;
        while (offset < this->myLength) {
            uint32_t remaining = this->myLength - offset;
            uint32_t curLen = (remaining >= this->tileLength) ? this->tileLength : remaining;
            CopyIn(offset, curLen);
            Compute(curLen);
            CopyOut(offset, curLen);
            offset += curLen;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(DTYPE_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
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
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Mins(zLocal, xLocal, (DTYPE_X)1.0f, len);
        AscendC::Maxs(zLocal, zLocal, (DTYPE_X)0.0f, len);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = len * sizeof(DTYPE_Z);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t myLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_group_norm_scale_max_pool_clamp_custom(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelClamp op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
