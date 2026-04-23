
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_PAIRS = 2048;
constexpr uint32_t TILE_ELEMENTS = TILE_PAIRS * 2;
constexpr uint32_t BATCHES_PER_BLOCK = 4;

class KernelMatmulMaxPoolSumScale {
public:
    __aicore__ inline KernelMatmulMaxPoolSumScale() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalBatches, uint32_t features)
    {
        this->totalBatches = totalBatches;
        this->features = features;

        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->startBatch = blockIdx * BATCHES_PER_BLOCK;
        this->endBatch = this->startBatch + BATCHES_PER_BLOCK;
        if (this->endBatch > totalBatches) {
            this->endBatch = totalBatches;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, totalBatches * features);
        yGm.SetGlobalBuffer((__gm__ float*)y, totalBatches);

        pipe.InitBuffer(inQueueEvens, BUFFER_NUM, TILE_PAIRS * sizeof(float));
        pipe.InitBuffer(inQueueOdds, BUFFER_NUM, TILE_PAIRS * sizeof(float));
        pipe.InitBuffer(maxBuf, TILE_PAIRS * sizeof(float));
        pipe.InitBuffer(reduceBuf, 8 * 1024);
        pipe.InitBuffer(reduceResBuf, 32);
        pipe.InitBuffer(outQueueResult, 1, 32);
    }

    __aicore__ inline void Process()
    {
        uint32_t tilesPerBatch = features / TILE_ELEMENTS;

        AscendC::LocalTensor<float> resultLocal = outQueueResult.AllocTensor<float>();

        for (uint32_t b = 0; b < BATCHES_PER_BLOCK; b++) {
            uint32_t batch = startBatch + b;
            float batchSum = 0.0f;

            if (batch < endBatch) {
                for (uint32_t tile = 0; tile < tilesPerBatch; tile++) {
                    CopyIn(batch, tile);
                    batchSum += ComputeAndReduce();
                }
            }

            resultLocal.SetValue(b, batchSum);
        }

        outQueueResult.EnQue(resultLocal);
        AscendC::LocalTensor<float> r = outQueueResult.DeQue<float>();

        uint32_t writeCount = endBatch - startBatch;
        if (writeCount > 0) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = writeCount * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(yGm[startBatch], r, copyParams);
        }
        outQueueResult.FreeTensor(r);
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch, uint32_t tile)
    {
        AscendC::LocalTensor<float> evensLocal = inQueueEvens.AllocTensor<float>();
        AscendC::LocalTensor<float> oddsLocal = inQueueOdds.AllocTensor<float>();

        uint32_t gmOffset = batch * features + tile * TILE_ELEMENTS;

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = TILE_PAIRS;
        copyParams.blockLen = sizeof(float);
        copyParams.srcStride = sizeof(float);
        copyParams.dstStride = 0;

        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;

        AscendC::DataCopyPad(evensLocal, xGm[gmOffset], copyParams, padParams);
        AscendC::DataCopyPad(oddsLocal, xGm[gmOffset + 1], copyParams, padParams);

        inQueueEvens.EnQue(evensLocal);
        inQueueOdds.EnQue(oddsLocal);
    }

    __aicore__ inline float ComputeAndReduce()
    {
        AscendC::LocalTensor<float> evensLocal = inQueueEvens.DeQue<float>();
        AscendC::LocalTensor<float> oddsLocal = inQueueOdds.DeQue<float>();
        AscendC::LocalTensor<float> maxLocal = maxBuf.Get<float>();

        AscendC::Max(maxLocal, evensLocal, oddsLocal, TILE_PAIRS);

        inQueueEvens.FreeTensor(evensLocal);
        inQueueOdds.FreeTensor(oddsLocal);

        AscendC::LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> reduceRes = reduceResBuf.Get<float>();
        AscendC::ReduceSum<float>(reduceRes, maxLocal, reduceLocal, static_cast<int32_t>(TILE_PAIRS));

        return reduceRes.GetValue(0);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueEvens;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueOdds;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueResult;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceResBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t totalBatches;
    uint32_t features;
    uint32_t startBatch;
    uint32_t endBatch;
};

extern "C" __global__ __aicore__ void matmul_max_pool_sum_scale_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulMaxPoolSumScale op;
    op.Init(x, y, tiling_data.totalBatches, tiling_data.features);
    op.Process();
}
