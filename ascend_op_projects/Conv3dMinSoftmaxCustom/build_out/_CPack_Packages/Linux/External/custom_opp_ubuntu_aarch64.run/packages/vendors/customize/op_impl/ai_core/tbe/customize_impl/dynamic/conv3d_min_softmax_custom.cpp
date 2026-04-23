
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 1;

class KernelConv3dMinSoftmax {
public:
    __aicore__ inline KernelConv3dMinSoftmax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
        uint32_t N_, uint32_t C_, uint32_t D_, uint32_t H_, uint32_t W_,
        uint32_t totalSlices, uint32_t slicesPerCore)
    {
        this->N = N_;
        this->C = C_;
        this->D = D_;
        this->H = H_;
        this->W = W_;
        this->rowSize = ((W_ + 7) / 8) * 8;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t s0 = blockIdx * slicesPerCore;
        uint32_t s1 = s0 + slicesPerCore;
        if (s0 > totalSlices) s0 = totalSlices;
        if (s1 > totalSlices) s1 = totalSlices;
        this->startSlice = s0;
        this->endSlice = s1;

        uint64_t xSize = (uint64_t)N * C * D * H * W;
        uint64_t ySize = (uint64_t)N * C * H * W;

        xGm.SetGlobalBuffer((__gm__ float*)x, xSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, ySize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, C * D * rowSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, C * rowSize * sizeof(float));
        pipe.InitBuffer(minBuf, C * rowSize * sizeof(float));
        pipe.InitBuffer(maxBuf, rowSize * sizeof(float));
        pipe.InitBuffer(sumBuf, rowSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t slice = startSlice; slice < endSlice; slice++) {
            uint32_t n = slice / H;
            uint32_t h = slice % H;
            CopyIn(n, h);
            Compute();
            CopyOut(n, h);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t n, uint32_t h)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        uint64_t base = (uint64_t)n * C * D * H * W + (uint64_t)h * W;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = (uint16_t)(C * D);
        copyParams.blockLen = (uint32_t)(W * sizeof(float));
        copyParams.srcStride = (uint32_t)((uint64_t)(H - 1) * W * sizeof(float));
        copyParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (uint8_t)(rowSize - W);
        padParams.paddingValue = 0.0f;

        AscendC::DataCopyPad(xLocal, xGm[base], copyParams, padParams);

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> minLocal = minBuf.Get<float>();
        AscendC::LocalTensor<float> maxLocal = maxBuf.Get<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();

        for (uint32_t c = 0; c < C; c++) {
            uint32_t cBase = c * D * rowSize;
            uint32_t minBase = c * rowSize;

            AscendC::Adds<float>(minLocal[minBase], xLocal[cBase], 0.0f, rowSize);
            for (uint32_t d = 1; d < D; d++) {
                AscendC::Min<float>(minLocal[minBase], minLocal[minBase], xLocal[cBase + d * rowSize], rowSize);
            }
        }

        AscendC::Adds<float>(maxLocal, minLocal, 0.0f, rowSize);
        for (uint32_t c = 1; c < C; c++) {
            AscendC::Max<float>(maxLocal, maxLocal, minLocal[c * rowSize], rowSize);
        }

        AscendC::BinaryRepeatParams bParams;
        bParams.dstBlkStride = 1;
        bParams.src0BlkStride = 1;
        bParams.src1BlkStride = 1;
        bParams.dstRepStride = (uint8_t)(rowSize / 8);
        bParams.src0RepStride = (uint8_t)(rowSize / 8);
        bParams.src1RepStride = 0;

        AscendC::Sub<float>(minLocal, minLocal, maxLocal, (uint64_t)rowSize, (uint8_t)C, bParams);

        AscendC::Exp<float>(minLocal, minLocal, C * rowSize);

        AscendC::Adds<float>(sumLocal, minLocal, 0.0f, rowSize);
        for (uint32_t c = 1; c < C; c++) {
            AscendC::Add<float>(sumLocal, sumLocal, minLocal[c * rowSize], rowSize);
        }

        AscendC::Div<float>(yLocal, minLocal, sumLocal, (uint64_t)rowSize, (uint8_t)C, bParams);

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t n, uint32_t h)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();

        uint64_t base = (uint64_t)n * C * H * W + (uint64_t)h * W;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = (uint16_t)C;
        copyParams.blockLen = (uint32_t)(W * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = (uint32_t)((uint64_t)(H - 1) * W * sizeof(float));

        AscendC::DataCopyPad(yGm[base], yLocal, copyParams);

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> minBuf, maxBuf, sumBuf;
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t N, C, D, H, W;
    uint32_t rowSize;
    uint32_t startSlice, endSlice;
};

extern "C" __global__ __aicore__ void conv3d_min_softmax_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMinSoftmax op;
    op.Init(x, y, tiling_data.N, tiling_data.C, tiling_data.D, tiling_data.H, tiling_data.W,
        tiling_data.totalSlices, tiling_data.slicesPerCore);
    op.Process();
}
