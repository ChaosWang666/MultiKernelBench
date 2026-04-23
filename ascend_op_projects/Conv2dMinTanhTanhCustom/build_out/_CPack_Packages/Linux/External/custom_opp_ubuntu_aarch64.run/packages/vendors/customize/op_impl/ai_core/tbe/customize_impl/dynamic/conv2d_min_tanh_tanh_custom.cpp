
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_HW = 256;

class KernelConv2dMinTanhTanh {
public:
    __aicore__ inline KernelConv2dMinTanhTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchPerBlock,
                                 uint32_t channels, uint32_t hw, uint32_t totalBatch)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startBatch = blockIdx * batchPerBlock;
        uint32_t endBatch = startBatch + batchPerBlock;
        if (endBatch > totalBatch) {
            endBatch = totalBatch;
        }
        this->myBatchCount = (endBatch > startBatch) ? (endBatch - startBatch) : 0;
        this->channels = channels;
        this->hw = hw;
        this->numTiles = (hw + TILE_HW - 1) / TILE_HW;

        xGm.SetGlobalBuffer((__gm__ float *)x + startBatch * channels * hw,
                            this->myBatchCount * channels * hw);
        yGm.SetGlobalBuffer((__gm__ float *)y + startBatch * hw,
                            this->myBatchCount * hw);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, channels * TILE_HW * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_HW * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t n = 0; n < myBatchCount; n++) {
            for (uint32_t t = 0; t < numTiles; t++) {
                uint32_t startHW = t * TILE_HW;
                uint32_t remaining = hw - startHW;
                uint32_t actualTile = (remaining < TILE_HW) ? remaining : TILE_HW;
                CopyIn(n, startHW, actualTile);
                Compute(actualTile);
                CopyOut(n, startHW, actualTile);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t n, uint32_t startHW, uint32_t actualTile)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(actualTile * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        uint64_t baseOffset = (uint64_t)n * channels * hw + startHW;
        for (uint32_t c = 0; c < channels; c++) {
            AscendC::DataCopyPad(xLocal[c * TILE_HW],
                                 xGm[baseOffset + c * hw],
                                 copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t actualTile)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        AscendC::Adds<float>(yLocal, xLocal, 0.0f, actualTile);

        for (uint32_t c = 1; c < channels; c++) {
            AscendC::Min<float>(yLocal, yLocal, xLocal[c * TILE_HW], actualTile);
        }

        AscendC::Tanh<float>(yLocal, yLocal, actualTile);
        AscendC::Tanh<float>(yLocal, yLocal, actualTile);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t n, uint32_t startHW, uint32_t actualTile)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(actualTile * sizeof(float));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::DataCopyPad(yGm[(uint64_t)n * hw + startHW], yLocal, copyParams);

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t myBatchCount;
    uint32_t channels;
    uint32_t hw;
    uint32_t numTiles;
};

extern "C" __global__ __aicore__ void conv2d_min_tanh_tanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dMinTanhTanh op;
    op.Init(x, y, tiling_data.batchPerBlock, tiling_data.channels,
            tiling_data.hw, tiling_data.totalBatch);
    op.Process();
}
