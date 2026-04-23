
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGAP {
public:
    __aicore__ inline KernelGAP() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t numGroups, uint32_t groupSize,
                                 uint32_t tileLength, uint32_t groupsPerBlock)
    {
        this->numGroups = numGroups;
        this->groupSize = groupSize;
        this->tileLength = tileLength;
        this->groupsPerBlock = groupsPerBlock;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->groupStart = blockIdx * groupsPerBlock;
        uint32_t tmpEnd = this->groupStart + groupsPerBlock;
        if (this->groupStart > numGroups) this->groupStart = numGroups;
        this->groupEnd = (tmpEnd < numGroups) ? tmpEnd : numGroups;

        xGm.SetGlobalBuffer((__gm__ float*)x, numGroups * groupSize);
        yGm.SetGlobalBuffer((__gm__ float*)y, numGroups);

        uint32_t outBufSize = groupsPerBlock * sizeof(float);
        if (outBufSize < 32) outBufSize = 32;
        outBufSize = ((outBufSize + 31) / 32) * 32;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, 1, outBufSize);
        pipe.InitBuffer(sumBuf, 32);
        pipe.InitBuffer(workBuf, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (groupStart >= groupEnd) return;
        uint32_t numGroupsInBlock = groupEnd - groupStart;

        AscendC::LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();

        for (uint32_t g = groupStart; g < groupEnd; g++) {
            float mean = ComputeGroupMean(g);
            outLocal.SetValue(g - groupStart, mean);
        }

        outQueueY.EnQue(outLocal);
        AscendC::LocalTensor<float> outOut = outQueueY.DeQue<float>();

        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(numGroupsInBlock * sizeof(float));
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        AscendC::DataCopyPad(yGm[groupStart], outOut, copyOutParams);
        outQueueY.FreeTensor(outOut);
    }

private:
    __aicore__ inline float ComputeGroupMean(uint32_t groupIdx)
    {
        uint64_t offset = (uint64_t)groupIdx * (uint64_t)groupSize;
        uint32_t numTiles = (groupSize + tileLength - 1) / tileLength;

        float totalSum = 0.0f;

        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t start = t * tileLength;
            uint32_t curLen = (start + tileLength <= groupSize) ? tileLength : (groupSize - start);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(curLen * sizeof(float));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;

            AscendC::DataCopyPadExtParams<float> padParams;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0.0f;

            AscendC::DataCopyPad(xLocal, xGm[offset + start], copyParams, padParams);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
            AscendC::LocalTensor<float> workLocal = workBuf.Get<float>();

            AscendC::ReduceSum<float, true>(sumLocal, xIn, workLocal, static_cast<int32_t>(curLen));
            float partial = sumLocal.GetValue(0);
            totalSum += partial;

            inQueueX.FreeTensor(xIn);
        }

        return totalSum / static_cast<float>(groupSize);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t numGroups;
    uint32_t groupSize;
    uint32_t tileLength;
    uint32_t groupsPerBlock;
    uint32_t groupStart;
    uint32_t groupEnd;
};

extern "C" __global__ __aicore__ void conv_transpose3d_scale_batch_norm_global_avg_pool_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGAP op;
    op.Init(x, y, tiling_data.numGroups, tiling_data.groupSize,
            tiling_data.tileLength, tiling_data.groupsPerBlock);
    op.Process();
}
