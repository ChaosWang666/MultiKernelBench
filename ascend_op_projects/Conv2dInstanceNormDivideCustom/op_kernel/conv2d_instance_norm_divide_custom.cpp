
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelInstanceNormDivide {
public:
    __aicore__ inline KernelInstanceNormDivide() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalInstances, uint32_t hw,
                                uint32_t tileSize, uint32_t instancesPerCore,
                                float eps, float divideBy, AscendC::TPipe* pipe)
    {
        this->totalInstances = totalInstances;
        this->hw = hw;
        this->tileSize = tileSize;
        this->instancesPerCore = instancesPerCore;
        this->eps = eps;
        this->divideBy = divideBy;
        this->invHW = 1.0f / static_cast<float>(hw);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->startInst = blockIdx * instancesPerCore;
        uint32_t endCandidate = this->startInst + instancesPerCore;
        this->endInst = (endCandidate > totalInstances) ? totalInstances : endCandidate;

        uint64_t totalElems = static_cast<uint64_t>(totalInstances) * static_cast<uint64_t>(hw);
        xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);

        pipe->InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe->InitBuffer(workBuf, tileSize * sizeof(float));
        pipe->InitBuffer(reduceBuf, 4 * 1024);
        pipe->InitBuffer(scalarBuf, 64);
    }

    __aicore__ inline void Process()
    {
        if (startInst >= endInst) {
            return;
        }
        for (uint32_t inst = startInst; inst < endInst; inst++) {
            ProcessInstance(inst);
        }
    }

private:
    __aicore__ inline void ProcessInstance(uint32_t inst)
    {
        uint64_t offset = static_cast<uint64_t>(inst) * static_cast<uint64_t>(hw);
        uint32_t numTiles = (hw + tileSize - 1) / tileSize;

        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();
        AscendC::LocalTensor<float> work = workBuf.Get<float>();

        float totalSum = 0.0f;
        float totalSumSq = 0.0f;

        // Pass 1: accumulate sum and sum of squares
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t tileOff = t * tileSize;
            uint32_t curTile = (tileOff + tileSize <= hw) ? tileSize : (hw - tileOff);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curTile * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(xLocal, xGm[offset + tileOff], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();

            AscendC::ReduceSum<float, true>(scalar, xIn, reduceTmp, static_cast<int32_t>(curTile));
            totalSum += scalar.GetValue(0);

            AscendC::Mul<float>(work, xIn, xIn, curTile);
            AscendC::ReduceSum<float, true>(scalar, work, reduceTmp, static_cast<int32_t>(curTile));
            totalSumSq += scalar.GetValue(0);

            inQueueX.FreeTensor(xIn);
        }

        float mean = totalSum * invHW;
        float var = totalSumSq * invHW - mean * mean;
        if (var < 0.0f) {
            var = 0.0f;
        }

        AscendC::Duplicate<float>(scalar, var + eps, 8);
        AscendC::Sqrt<float>(scalar, scalar, 8);
        float stdVal = scalar.GetValue(0);
        float rstd = 1.0f / (stdVal * divideBy);
        float negMeanRstd = -mean * rstd;

        // Pass 2: normalize and write back
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t tileOff = t * tileSize;
            uint32_t curTile = (tileOff + tileSize <= hw) ? tileSize : (hw - tileOff);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = curTile * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            AscendC::DataCopyPad(xLocal, xGm[offset + tileOff], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            AscendC::Muls<float>(yLocal, xIn, rstd, curTile);
            AscendC::Adds<float>(yLocal, yLocal, negMeanRstd, curTile);

            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xIn);

            AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
            AscendC::DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = curTile * sizeof(float);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            copyOutParams.rsv = 0;
            AscendC::DataCopyPad(yGm[offset + tileOff], yOut, copyOutParams);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    uint32_t totalInstances;
    uint32_t hw;
    uint32_t tileSize;
    uint32_t instancesPerCore;
    uint32_t startInst;
    uint32_t endInst;
    float eps;
    float divideBy;
    float invHW;
};

extern "C" __global__ __aicore__ void conv2d_instance_norm_divide_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelInstanceNormDivide op;
    op.Init(x, y,
            tiling_data.totalInstances, tiling_data.hw,
            tiling_data.tileSize, tiling_data.instancesPerCore,
            tiling_data.eps, tiling_data.divideBy, &pipe);
    op.Process();
}
