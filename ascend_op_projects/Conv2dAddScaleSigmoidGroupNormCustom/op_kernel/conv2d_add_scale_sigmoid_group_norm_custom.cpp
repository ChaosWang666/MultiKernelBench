
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LEN = 2048;
constexpr uint32_t MAX_CHANNELS = 64;

class KernelFusedGN {
public:
    __aicore__ inline KernelFusedGN() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR scale,
                                GM_ADDR gamma, GM_ADDR beta, GM_ADDR z,
                                uint32_t batchSize, uint32_t channels,
                                uint32_t hw, uint32_t numGroups,
                                uint32_t channelsPerGroup, float eps)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->hw = hw;
        this->numGroups = numGroups;
        this->channelsPerGroup = channelsPerGroup;
        this->eps = eps;

        uint32_t totalPairs = batchSize * numGroups;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t pairsPerBlock = (totalPairs + blockNum - 1) / blockNum;
        this->startPair = blockIdx * pairsPerBlock;
        uint32_t endPair = this->startPair + pairsPerBlock;
        if (endPair > totalPairs) endPair = totalPairs;
        this->numPairs = (this->startPair < totalPairs) ? (endPair - this->startPair) : 0;

        xGm.SetGlobalBuffer((__gm__ float *)x);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, channels);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, channels);
        gammaGm.SetGlobalBuffer((__gm__ float *)gamma, channels);
        betaGm.SetGlobalBuffer((__gm__ float *)beta, channels);
        zGm.SetGlobalBuffer((__gm__ float *)z);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LEN * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LEN * sizeof(float));
        pipe.InitBuffer(paramsBuf, 4 * MAX_CHANNELS * sizeof(float));
        pipe.InitBuffer(sigBuf, TILE_LEN * sizeof(float));
        pipe.InitBuffer(onesBuf, TILE_LEN * sizeof(float));
        pipe.InitBuffer(reduceBuf, 32 * 1024);
        pipe.InitBuffer(scalarBuf, 32);
    }

    __aicore__ inline void Process()
    {
        if (numPairs == 0) return;

        AscendC::LocalTensor<float> paramsLocal = paramsBuf.Get<float>();
        AscendC::LocalTensor<float> onesLocal = onesBuf.Get<float>();

        AscendC::Duplicate<float>(onesLocal, 1.0f, TILE_LEN);

        AscendC::DataCopyExtParams cp;
        cp.blockCount = 1;
        cp.blockLen = channels * sizeof(float);
        cp.srcStride = 0;
        cp.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> pp;
        pp.isPad = false;
        pp.leftPadding = 0;
        pp.rightPadding = 0;
        pp.paddingValue = 0.0f;

        AscendC::DataCopyPad(paramsLocal[0], biasGm, cp, pp);
        AscendC::DataCopyPad(paramsLocal[MAX_CHANNELS], scaleGm, cp, pp);
        AscendC::DataCopyPad(paramsLocal[2 * MAX_CHANNELS], gammaGm, cp, pp);
        AscendC::DataCopyPad(paramsLocal[3 * MAX_CHANNELS], betaGm, cp, pp);

        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t i = 0; i < numPairs; i++) {
            uint32_t pairIdx = startPair + i;
            uint32_t n = pairIdx / numGroups;
            uint32_t g = pairIdx % numGroups;
            ProcessPair(n, g);
        }
    }

private:
    __aicore__ inline void ProcessPair(uint32_t n, uint32_t g)
    {
        AscendC::LocalTensor<float> paramsLocal = paramsBuf.Get<float>();
        AscendC::LocalTensor<float> sigLocal = sigBuf.Get<float>();
        AscendC::LocalTensor<float> onesLocal = onesBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();

        uint32_t groupSize = channelsPerGroup * hw;
        uint32_t baseChannel = g * channelsPerGroup;
        uint64_t sampleOffset = (uint64_t)n * channels * hw + (uint64_t)baseChannel * hw;

        float sum = 0.0f;
        float sumsq = 0.0f;

        // Pass 1: accumulate sum and sum of squares of sigmoid output
        for (uint32_t c_off = 0; c_off < channelsPerGroup; c_off++) {
            float b = paramsLocal.GetValue(baseChannel + c_off);
            float s = paramsLocal.GetValue(MAX_CHANNELS + baseChannel + c_off);
            uint64_t channelOffset = sampleOffset + (uint64_t)c_off * hw;

            for (uint32_t tileStart = 0; tileStart < hw; tileStart += TILE_LEN) {
                uint32_t tileLen = (tileStart + TILE_LEN <= hw) ? TILE_LEN : (hw - tileStart);

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = tileLen * sizeof(float);
                cp.srcStride = 0;
                cp.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> pp;
                pp.isPad = false;
                pp.leftPadding = 0;
                pp.rightPadding = 0;
                pp.paddingValue = 0.0f;
                AscendC::DataCopyPad(xLocal, xGm[channelOffset + tileStart], cp, pp);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();

                // sigLocal = sigmoid((x + b) * s) = 1 / (1 + exp(-(x+b)*s))
                AscendC::Adds<float>(sigLocal, xIn, b, tileLen);
                AscendC::Muls<float>(sigLocal, sigLocal, -s, tileLen);
                AscendC::Exp<float>(sigLocal, sigLocal, tileLen);
                AscendC::Adds<float>(sigLocal, sigLocal, 1.0f, tileLen);
                AscendC::Div<float>(sigLocal, onesLocal, sigLocal, tileLen);

                inQueueX.FreeTensor(xIn);

                AscendC::ReduceSum<float, true>(scalarLocal, sigLocal, reduceTmp, tileLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                sum += scalarLocal.GetValue(0);

                AscendC::Mul<float>(sigLocal, sigLocal, sigLocal, tileLen);
                AscendC::ReduceSum<float, true>(scalarLocal, sigLocal, reduceTmp, tileLen);
                AscendC::PipeBarrier<PIPE_ALL>();
                sumsq += scalarLocal.GetValue(0);
            }
        }

        float mean = sum / (float)groupSize;
        float var = sumsq / (float)groupSize - mean * mean;
        float varPlusEps = var + eps;

        // Compute invStd using tensor Sqrt
        AscendC::Duplicate<float>(scalarLocal, varPlusEps, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sqrt<float>(scalarLocal, scalarLocal, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        float stdVal = scalarLocal.GetValue(0);
        float invStd = 1.0f / stdVal;

        // Pass 2: recompute sigmoid and normalize
        for (uint32_t c_off = 0; c_off < channelsPerGroup; c_off++) {
            float b = paramsLocal.GetValue(baseChannel + c_off);
            float s = paramsLocal.GetValue(MAX_CHANNELS + baseChannel + c_off);
            float gam = paramsLocal.GetValue(2 * MAX_CHANNELS + baseChannel + c_off);
            float bet = paramsLocal.GetValue(3 * MAX_CHANNELS + baseChannel + c_off);

            float scale_factor = invStd * gam;
            float bias_factor = bet - mean * scale_factor;

            uint64_t channelOffset = sampleOffset + (uint64_t)c_off * hw;

            for (uint32_t tileStart = 0; tileStart < hw; tileStart += TILE_LEN) {
                uint32_t tileLen = (tileStart + TILE_LEN <= hw) ? TILE_LEN : (hw - tileStart);

                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = tileLen * sizeof(float);
                cp.srcStride = 0;
                cp.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> pp;
                pp.isPad = false;
                pp.leftPadding = 0;
                pp.rightPadding = 0;
                pp.paddingValue = 0.0f;
                AscendC::DataCopyPad(xLocal, xGm[channelOffset + tileStart], cp, pp);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();

                // Recompute sigmoid
                AscendC::Adds<float>(sigLocal, xIn, b, tileLen);
                AscendC::Muls<float>(sigLocal, sigLocal, -s, tileLen);
                AscendC::Exp<float>(sigLocal, sigLocal, tileLen);
                AscendC::Adds<float>(sigLocal, sigLocal, 1.0f, tileLen);
                AscendC::Div<float>(sigLocal, onesLocal, sigLocal, tileLen);

                inQueueX.FreeTensor(xIn);

                // z = sig * scale_factor + bias_factor
                AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
                AscendC::Muls<float>(sigLocal, sigLocal, scale_factor, tileLen);
                AscendC::Adds<float>(zLocal, sigLocal, bias_factor, tileLen);
                outQueueZ.EnQue(zLocal);

                AscendC::LocalTensor<float> zOut = outQueueZ.DeQue<float>();
                AscendC::DataCopyExtParams cpOut;
                cpOut.blockCount = 1;
                cpOut.blockLen = tileLen * sizeof(float);
                cpOut.srcStride = 0;
                cpOut.dstStride = 0;
                AscendC::DataCopyPad(zGm[channelOffset + tileStart], zOut, cpOut);
                outQueueZ.FreeTensor(zOut);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> paramsBuf, sigBuf, onesBuf, reduceBuf, scalarBuf;
    AscendC::GlobalTensor<float> xGm, biasGm, scaleGm, gammaGm, betaGm, zGm;
    uint32_t batchSize, channels, hw, numGroups, channelsPerGroup;
    uint32_t startPair, numPairs;
    float eps;
};

extern "C" __global__ __aicore__ void conv2d_add_scale_sigmoid_group_norm_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedGN op;
    op.Init(x, bias, scale, gamma, beta, z,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.hw, tiling_data.numGroups,
            tiling_data.channelsPerGroup, tiling_data.eps);
    op.Process();
}
