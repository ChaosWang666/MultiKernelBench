
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFusedOp {
public:
    __aicore__ inline KernelFusedOp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sumTensor, GM_ADDR y,
                                uint32_t totalRows, uint32_t channelSize,
                                uint32_t spatialSize, uint32_t rowTileSize)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t rowsPerCore = totalRows / blockNum;
        uint32_t tailRows = totalRows % blockNum;

        if (blockIdx < tailRows) {
            this->rowsThisCore = rowsPerCore + 1;
            this->startRow = blockIdx * (rowsPerCore + 1);
        } else {
            this->rowsThisCore = rowsPerCore;
            this->startRow = tailRows * (rowsPerCore + 1) + (blockIdx - tailRows) * rowsPerCore;
        }

        this->channelSize = channelSize;
        this->spatialSize = spatialSize;
        this->rowTileSize = rowTileSize;

        uint64_t totalElems = (uint64_t)totalRows * (uint64_t)spatialSize;
        xGm.SetGlobalBuffer((__gm__ float*)x, totalElems);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalElems);
        sumGm.SetGlobalBuffer((__gm__ float*)sumTensor, channelSize);

        pipe.InitBuffer(inQueue, BUFFER_NUM, rowTileSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, rowTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, rowTileSize * sizeof(float));

        uint32_t sumBufBytes = ((channelSize * (uint32_t)sizeof(float) + 31U) / 32U) * 32U;
        pipe.InitBuffer(sumBuf, sumBufBytes);
    }

    __aicore__ inline void Process()
    {
        if (rowsThisCore == 0) return;

        // Load sum_tensor into UB once
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::DataCopyExtParams sumCopyParams;
        sumCopyParams.blockCount = 1;
        sumCopyParams.blockLen = channelSize * (uint32_t)sizeof(float);
        sumCopyParams.srcStride = 0;
        sumCopyParams.dstStride = 0;
        sumCopyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> sumPadParams;
        sumPadParams.isPad = false;
        sumPadParams.leftPadding = 0;
        sumPadParams.rightPadding = 0;
        sumPadParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(sumLocal, sumGm, sumCopyParams, sumPadParams);

        // MTE2 -> Scalar sync for GetValue
        auto eventIdMTE2ToS = AscendC::GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIdMTE2ToS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIdMTE2ToS);

        for (uint32_t r = 0; r < rowsThisCore; r++) {
            uint32_t globalRow = startRow + r;
            uint32_t channel = globalRow % channelSize;
            float scalar = sumLocal.GetValue(channel);

            uint32_t processedInRow = 0;
            while (processedInRow < spatialSize) {
                uint32_t remaining = spatialSize - processedInRow;
                uint32_t curTileSize = (remaining < rowTileSize) ? remaining : rowTileSize;
                uint64_t gmOffset = (uint64_t)globalRow * (uint64_t)spatialSize + (uint64_t)processedInRow;
                ProcessTile(gmOffset, curTileSize, scalar);
                processedInRow += curTileSize;
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(uint64_t offset, uint32_t tileSize, float scalar)
    {
        // CopyIn
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = tileSize * (uint32_t)sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(xLocal, xGm[offset], copyInParams, padParams);
        inQueue.EnQue(xLocal);

        // Compute
        AscendC::LocalTensor<float> xIn = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp = tmpBuf.Get<float>();

        // 1. LeakyReLU: y = 0.8 * max(x, 0) + 0.2 * x
        AscendC::Maxs(tmp, xIn, 0.0f, (int32_t)tileSize);
        AscendC::Muls(tmp, tmp, 0.8f, (int32_t)tileSize);
        AscendC::Muls(yLocal, xIn, 0.2f, (int32_t)tileSize);
        AscendC::Add(yLocal, yLocal, tmp, (int32_t)tileSize);

        // 2. Add per-channel scalar
        AscendC::Adds(yLocal, yLocal, scalar, (int32_t)tileSize);

        // 3. Clamp to [-1, 1]
        AscendC::Maxs(yLocal, yLocal, -1.0f, (int32_t)tileSize);
        AscendC::Mins(yLocal, yLocal, 1.0f, (int32_t)tileSize);

        // 4. GELU (tanh approximation)
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float SQRT_2_OVER_PI = 0.7978845608028654f;
        const float GELU_COEFF = 0.044715f;

        AscendC::Mul(tmp, yLocal, yLocal, (int32_t)tileSize);          // x^2
        AscendC::Mul(tmp, tmp, yLocal, (int32_t)tileSize);             // x^3
        AscendC::Muls(tmp, tmp, GELU_COEFF, (int32_t)tileSize);        // 0.044715 * x^3
        AscendC::Add(tmp, tmp, yLocal, (int32_t)tileSize);             // x + 0.044715*x^3
        AscendC::Muls(tmp, tmp, SQRT_2_OVER_PI, (int32_t)tileSize);    // inner
        AscendC::Tanh(tmp, tmp, (int32_t)tileSize);                    // tanh(inner)
        AscendC::Adds(tmp, tmp, 1.0f, (int32_t)tileSize);              // 1 + tanh(inner)
        AscendC::Mul(yLocal, yLocal, tmp, (int32_t)tileSize);          // x * (1 + tanh(inner))
        AscendC::Muls(yLocal, yLocal, 0.5f, (int32_t)tileSize);        // 0.5 * x * (1 + tanh(inner))

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xIn);

        // CopyOut
        AscendC::LocalTensor<float> yOut = outQueue.DeQue<float>();
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = tileSize * (uint32_t)sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        copyOutParams.rsv = 0;
        AscendC::DataCopyPad(yGm[offset], yOut, copyOutParams);
        outQueue.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> sumGm;
    uint32_t startRow;
    uint32_t rowsThisCore;
    uint32_t channelSize;
    uint32_t spatialSize;
    uint32_t rowTileSize;
};

extern "C" __global__ __aicore__ void conv3d_leaky_relu_sum_clamp_gelu_custom(
    GM_ADDR x, GM_ADDR sum_tensor, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedOp op;
    op.Init(x, sum_tensor, y, tiling_data.totalRows, tiling_data.channelSize,
            tiling_data.spatialSize, tiling_data.rowTileSize);
    op.Process();
}
