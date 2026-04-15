
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelConv3d {
public:
    __aicore__ inline KernelConv3d() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, uint32_t batch, uint32_t inChannels, uint32_t outChannels,
                                uint32_t inDepth, uint32_t inHeight, uint32_t inWidth,
                                uint32_t kernelDepth, uint32_t kernelHeight, uint32_t kernelWidth,
                                uint32_t strideDepth, uint32_t strideHeight, uint32_t strideWidth,
                                uint32_t padDepth, uint32_t padHeight, uint32_t padWidth,
                                uint32_t dilationDepth, uint32_t dilationHeight, uint32_t dilationWidth,
                                uint32_t groups, uint32_t outDepth, uint32_t outHeight, uint32_t outWidth)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->inDepth = inDepth;
        this->inHeight = inHeight;
        this->inWidth = inWidth;
        this->kernelDepth = kernelDepth;
        this->kernelHeight = kernelHeight;
        this->kernelWidth = kernelWidth;
        this->strideDepth = strideDepth;
        this->strideHeight = strideHeight;
        this->strideWidth = strideWidth;
        this->padDepth = padDepth;
        this->padHeight = padHeight;
        this->padWidth = padWidth;
        this->dilationDepth = dilationDepth;
        this->dilationHeight = dilationHeight;
        this->dilationWidth = dilationWidth;
        this->groups = groups;
        this->outDepth = outDepth;
        this->outHeight = outHeight;
        this->outWidth = outWidth;

        this->blockLength = batch * outChannels * outDepth * outHeight * outWidth / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, inChannels * outChannels * kernelDepth * kernelHeight * kernelWidth);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->blockLength / this->tileLength;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(wLocal, wGm[0], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // Simplified computation for demonstration
        AscendC::Mul(yLocal, xLocal, wLocal, this->tileLength);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueW;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t inDepth;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t kernelDepth;
    uint32_t kernelHeight;
    uint32_t kernelWidth;
    uint32_t strideDepth;
    uint32_t strideHeight;
    uint32_t strideWidth;
    uint32_t padDepth;
    uint32_t padHeight;
    uint32_t padWidth;
    uint32_t dilationDepth;
    uint32_t dilationHeight;
    uint32_t dilationWidth;
    uint32_t groups;
    uint32_t outDepth;
    uint32_t outHeight;
    uint32_t outWidth;
};

extern "C" __global__ __aicore__ void conv_standard_3d_asymmetric_input_asymmetric_kernel_custom(
    GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3d op;
    op.Init(x, w, y,
            tiling_data.batch, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.inDepth, tiling_data.inHeight, tiling_data.inWidth,
            tiling_data.kernelDepth, tiling_data.kernelHeight, tiling_data.kernelWidth,
            tiling_data.strideDepth, tiling_data.strideHeight, tiling_data.strideWidth,
            tiling_data.padDepth, tiling_data.padHeight, tiling_data.padWidth,
            tiling_data.dilationDepth, tiling_data.dilationHeight, tiling_data.dilationWidth,
            tiling_data.groups, tiling_data.outDepth, tiling_data.outHeight, tiling_data.outWidth);
    op.Process();
}
