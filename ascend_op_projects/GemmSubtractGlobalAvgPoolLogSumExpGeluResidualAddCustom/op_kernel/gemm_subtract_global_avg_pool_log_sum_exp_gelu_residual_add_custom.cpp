
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelOp {
public:
    __aicore__ inline KernelOp() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR x, GM_ADDR z,
                                 uint32_t batchSize, uint32_t nLen, uint32_t mLen,
                                 uint32_t rowsPerBlock, uint32_t tileSize)
    {
        this->nLen = nLen;
        this->mLen = mLen;
        this->tileSize = tileSize;
        this->invN = 1.0f / (float)nLen;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t startRow = blockIdx * rowsPerBlock;
        if (startRow >= batchSize) {
            this->rowsThisBlock = 0;
            this->skip = true;
            return;
        }
        this->rowsThisBlock = (startRow + rowsPerBlock <= batchSize)
                              ? rowsPerBlock : (batchSize - startRow);
        this->skip = false;

        yGm.SetGlobalBuffer((__gm__ float*)y + startRow * nLen, this->rowsThisBlock * nLen);
        xGm.SetGlobalBuffer((__gm__ float*)x + startRow * mLen, this->rowsThisBlock * mLen);
        zGm.SetGlobalBuffer((__gm__ float*)z + startRow * mLen, this->rowsThisBlock * mLen);

        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(reduceBuf, 2048);
        pipe.InitBuffer(scalarBuf, 256);
    }

    __aicore__ inline void Process()
    {
        if (this->skip || this->rowsThisBlock == 0) return;
        for (uint32_t r = 0; r < this->rowsThisBlock; r++) {
            float meanVal = ComputeMean(r);
            float gVal = ComputeGelu(meanVal);
            ResidualAdd(r, gVal);
        }
    }

private:
    __aicore__ inline float ComputeMean(uint32_t r)
    {
        float totalSum = 0.0f;
        uint32_t numTiles = (nLen + tileSize - 1) / tileSize;
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceBuf.Get<float>();
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t offset = t * tileSize;
            uint32_t count = (offset + tileSize <= nLen) ? tileSize : (nLen - offset);

            AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
            AscendC::DataCopy(yLocal, yGm[r * nLen + offset], count);
            inQueueY.EnQue(yLocal);

            AscendC::LocalTensor<float> yIn = inQueueY.DeQue<float>();
            AscendC::ReduceSum<float, true>(scalarLocal, yIn, reduceTmp, (int32_t)count);
            totalSum += scalarLocal.GetValue(0);
            inQueueY.FreeTensor(yIn);
        }
        return totalSum * invN;
    }

    __aicore__ inline float ComputeGelu(float v)
    {
        const float c0 = 0.7978845608028654f;
        const float c1 = 0.044715f;
        float v3 = v * v * v;
        float inner = c0 * (v + c1 * v3);

        AscendC::LocalTensor<float> s = scalarBuf.Get<float>();
        AscendC::Duplicate<float>(s, -2.0f * inner, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Exp<float>(s, s, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        float e = s.GetValue(0);

        float denom = 1.0f + e;
        float tanhVal = (1.0f - e) / denom;
        return 0.5f * v * (1.0f + tanhVal);
    }

    __aicore__ inline void ResidualAdd(uint32_t r, float gVal)
    {
        uint32_t numTiles = (mLen + tileSize - 1) / tileSize;
        for (uint32_t t = 0; t < numTiles; t++) {
            uint32_t offset = t * tileSize;
            uint32_t count = (offset + tileSize <= mLen) ? tileSize : (mLen - offset);

            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[r * mLen + offset], count);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> zLocal = outQueue.AllocTensor<float>();
            AscendC::Adds<float>(zLocal, xIn, gVal, count);
            outQueue.EnQue<float>(zLocal);
            inQueueX.FreeTensor(xIn);

            AscendC::LocalTensor<float> zOut = outQueue.DeQue<float>();
            AscendC::DataCopy(zGm[r * mLen + offset], zOut, count);
            outQueue.FreeTensor(zOut);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> yGm, xGm, zGm;
    uint32_t nLen, mLen, tileSize, rowsThisBlock;
    float invN;
    bool skip;
};

extern "C" __global__ __aicore__ void gemm_subtract_global_avg_pool_log_sum_exp_gelu_residual_add_custom(
    GM_ADDR y, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelOp op;
    op.Init(y, x, z, tiling_data.batchSize, tiling_data.nLen, tiling_data.mLen,
            tiling_data.rowsPerBlock, tiling_data.tileSize);
    op.Process();
}
