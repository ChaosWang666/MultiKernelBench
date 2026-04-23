
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv3dMaxLseRelu {
public:
    __aicore__ inline KernelConv3dMaxLseRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batch, uint32_t channels,
                                 uint32_t spatial, uint32_t tileLen)
    {
        this->batch = batch;
        this->channels = channels;
        this->spatial = spatial;
        this->tileLen = tileLen;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t spatialPerCore = (spatial + blockNum - 1) / blockNum;
        spatialPerCore = (spatialPerCore + 7) / 8 * 8;

        uint32_t rawStart = blockIdx * spatialPerCore;
        uint32_t rawEnd = rawStart + spatialPerCore;
        if (rawStart > spatial) rawStart = spatial;
        if (rawEnd > spatial) rawEnd = spatial;
        this->myStart = rawStart;
        this->myEnd = rawEnd;

        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)batch * channels * spatial);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)batch * spatial);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(maxBuf, tileLen * sizeof(float));
        pipe.InitBuffer(sumBuf, tileLen * sizeof(float));
        pipe.InitBuffer(tmpBuf, tileLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (myEnd <= myStart) return;
        for (uint32_t b = 0; b < batch; b++) {
            for (uint32_t s = myStart; s < myEnd; s += tileLen) {
                uint32_t remaining = myEnd - s;
                uint32_t curTile = (remaining < tileLen) ? remaining : tileLen;
                ProcessTile(b, s, curTile);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t b, uint32_t s, uint32_t curTile)
    {
        AscendC::LocalTensor<float> maxLocal = maxBuf.Get<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        int32_t cnt = static_cast<int32_t>(curTile);

        AscendC::Duplicate<float>(maxLocal, -3.4e37f, cnt);

        for (uint32_t c = 0; c < channels; c++) {
            uint64_t gmOff = (uint64_t)b * channels * spatial + (uint64_t)c * spatial + s;

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams{1, (uint32_t)(curTile * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(xLocal, xGm[gmOff], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::Max<float>(maxLocal, maxLocal, xIn, cnt);
            inQueueX.FreeTensor(xIn);
        }

        AscendC::Duplicate<float>(sumLocal, 0.0f, cnt);

        for (uint32_t c = 0; c < channels; c++) {
            uint64_t gmOff = (uint64_t)b * channels * spatial + (uint64_t)c * spatial + s;

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams{1, (uint32_t)(curTile * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            AscendC::DataCopyPad(xLocal, xGm[gmOff], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::Sub<float>(tmpLocal, xIn, maxLocal, cnt);
            AscendC::Exp<float>(tmpLocal, tmpLocal, cnt);
            AscendC::Add<float>(sumLocal, sumLocal, tmpLocal, cnt);
            inQueueX.FreeTensor(xIn);
        }

        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Ln<float>(sumLocal, sumLocal, cnt);
        AscendC::Add<float>(yLocal, sumLocal, maxLocal, cnt);
        AscendC::Relu<float>(yLocal, yLocal, cnt);
        outQueueY.EnQue(yLocal);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        uint64_t outOff = (uint64_t)b * spatial + s;
        AscendC::DataCopyExtParams copyOutParams{1, (uint32_t)(curTile * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[outOff], yOut, copyOutParams);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t channels;
    uint32_t spatial;
    uint32_t tileLen;
    uint32_t myStart;
    uint32_t myEnd;
};

extern "C" __global__ __aicore__ void conv3d_max_log_sum_exp_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dMaxLseRelu op;
    op.Init(x, y, tiling_data.batch, tiling_data.channels, tiling_data.spatial, tiling_data.tileLen);
    op.Process();
}
