
#include "kernel_operator.h"

// KV-cached single-query attention kernel
// Q: (batch, 1, d_model), K: (batch, kv_len, d_model), V: (batch, kv_len, d_model)
// Output: (batch, 1, d_model)
// Each block handles one batch element.
// Algorithm (online softmax for numerical stability):
//   For each kv tile:
//     Compute dot products Q . K_tile^T -> partial scores
//     Track running max and sum for stable softmax
//     Accumulate weighted V
//   Normalize output

// We tile along kv_len. For each tile we:
// 1) Load a chunk of K (tileKv x dModel), compute scores = Q @ K_chunk^T / sqrt(d)
// 2) Do online softmax update
// 3) Load corresponding V chunk, accumulate output

constexpr int32_t BUFFER_NUM = 1;

class KernelKvCachedAttn {
public:
    __aicore__ inline KernelKvCachedAttn() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR output,
                                 GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t qLen, uint32_t kvLen, uint32_t dModel,
                                 uint32_t kvTileNum, uint32_t dTileNum)
    {
        this->batchSize = batchSize;
        this->qLen = qLen;
        this->kvLen = kvLen;
        this->dModel = dModel;
        this->kvTileNum = kvTileNum;
        this->dTileNum = dTileNum;

        uint32_t batchIdx = AscendC::GetBlockIdx();

        // Pointers offset by batch
        uint32_t qOffset = batchIdx * qLen * dModel;
        uint32_t kvOffset = batchIdx * kvLen * dModel;
        uint32_t outOffset = batchIdx * qLen * dModel;

        qGm.SetGlobalBuffer((__gm__ float*)q + qOffset, qLen * dModel);
        kGm.SetGlobalBuffer((__gm__ float*)k_cache + kvOffset, kvLen * dModel);
        vGm.SetGlobalBuffer((__gm__ float*)v_cache + kvOffset, kvLen * dModel);
        outGm.SetGlobalBuffer((__gm__ float*)output + outOffset, qLen * dModel);

        // Compute tile sizes
        // tileKv: number of kv positions per tile
        this->tileKv = kvLen / kvTileNum;
        // tileD: number of d dimensions per tile
        this->tileD = dModel / dTileNum;

        // Align tileKv and tileD to 8 (32 bytes / 4 bytes per float)
        // Assume they divide evenly for simplicity

        // Buffer sizes needed:
        // qBuf: tileD floats for a chunk of Q along d dimension
        // kBuf: tileKv * tileD floats for K chunk (but we process d tiles inside)
        // Actually let's simplify: for score computation we iterate d in tiles
        // scoreBuf: tileKv floats for accumulated scores
        // vBuf: tileKv floats for one d-column of V
        // outBuf: dModel floats for output accumulator

        // We'll use a simpler approach:
        // For each kv tile (tileKv rows of K):
        //   scores[tileKv] = 0
        //   For each d tile:
        //     Load Q_dtile[tileD], K_chunk_dtile[tileKv x tileD]
        //     scores += K_chunk_dtile @ Q_dtile (manual dot via Mul + ReduceSum per row, or element-wise)
        //   scores /= sqrt(d_model)
        //   Online softmax update with scores
        //   For each d tile of V:
        //     Load V_chunk_dtile[tileKv x tileD]
        //     output_dtile += scores . V_chunk_dtile (weighted sum)

        // Memory: we need tileD + tileKv*tileD + tileKv + tileKv*tileD + tileD
        // Let's allocate buffers

        uint32_t scoreBufSize = tileKv * sizeof(float);
        uint32_t qTileBufSize = tileD * sizeof(float);
        uint32_t kTileBufSize = tileKv * sizeof(float); // one d-element at a time across kv
        uint32_t tempBufSize = tileKv * sizeof(float);

        // Simpler approach: compute scores one element at a time along d
        // Too slow. Let's do: for each kv tile, load full K rows but tile along d.
        // Actually, the simplest correct approach for Ascend:
        // Process each kv position computing the dot product with Q.

        // Even simpler for correctness on AscendC:
        // For each kv-tile of size tileKv:
        //   Compute score[i] = sum_d Q[0,d]*K[i,d] for i in tile, using d-tiles
        //   Then softmax scores, then accumulate into output

        // Let's allocate pipe buffers
        // We'll manage memory manually with pipe

        pipe.InitBuffer(inQueueQ, 1, this->tileD * sizeof(float));
        pipe.InitBuffer(inQueueK, 1, this->tileKv * sizeof(float));
        pipe.InitBuffer(inQueueV, 1, this->tileKv * sizeof(float));
        pipe.InitBuffer(outQueueScore, 1, this->tileKv * sizeof(float));
        pipe.InitBuffer(outQueueOut, 1, this->dModel * sizeof(float));
        // temp buffers
        pipe.InitBuffer(tmpBuf1, 1, this->tileKv * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, this->tileKv * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Allocate output accumulator and zero it
        AscendC::LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        AscendC::Duplicate(outLocal, 0.0f, this->dModel);

        float globalMax = -1e30f;
        float globalSum = 0.0f;
        float scale = 1.0f / sqrtf((float)this->dModel);

        // For each kv tile
        for (uint32_t kvt = 0; kvt < this->kvTileNum; kvt++) {
            uint32_t kvBase = kvt * this->tileKv;

            // Compute scores for this kv tile
            AscendC::LocalTensor<float> scoreLocal = outQueueScore.AllocTensor<float>();
            AscendC::Duplicate(scoreLocal, 0.0f, this->tileKv);

            // Accumulate dot product tiled along d dimension
            for (uint32_t dt = 0; dt < this->dTileNum; dt++) {
                uint32_t dBase = dt * this->tileD;

                // Load Q tile: Q[0, dBase:dBase+tileD]
                AscendC::LocalTensor<float> qLocal = inQueueQ.AllocTensor<float>();
                AscendC::DataCopy(qLocal, qGm[dBase], this->tileD);
                AscendC::PipeBarrier<PIPE_ALL>();

                // For each kv position in tile, accumulate dot product
                for (uint32_t ki = 0; ki < this->tileKv; ki++) {
                    // Load K[kvBase+ki, dBase:dBase+tileD]
                    AscendC::LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
                    // K is stored as (kv_len, d_model), row = kvBase+ki, col start = dBase
                    AscendC::DataCopy(kLocal, kGm[(kvBase + ki) * this->dModel + dBase], this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // elementwise mul: kLocal = kLocal * qLocal
                    AscendC::LocalTensor<float> tmpLocal = tmpBuf1.AllocTensor<float>();
                    AscendC::Mul(tmpLocal, kLocal, qLocal, this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // reduce sum
                    float dotVal = 0.0f;
                    // Use ReduceSum if available, otherwise manual
                    AscendC::LocalTensor<float> sumTmp = tmpBuf2.AllocTensor<float>();
                    AscendC::ReduceSum(sumTmp, tmpLocal, tmpLocal, this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // sumTmp[0] has the result
                    dotVal = sumTmp.GetValue(0);

                    // Add to score
                    float curScore = scoreLocal.GetValue(ki);
                    scoreLocal.SetValue(ki, curScore + dotVal);

                    tmpBuf2.FreeTensor(sumTmp);
                    tmpBuf1.FreeTensor(tmpLocal);
                    inQueueK.FreeTensor(kLocal);
                }
                inQueueQ.FreeTensor(qLocal);
            }

            // Scale scores
            AscendC::Muls(scoreLocal, scoreLocal, scale, this->tileKv);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Online softmax update
            // Find local max
            AscendC::LocalTensor<float> tmpReduce = tmpBuf1.AllocTensor<float>();
            AscendC::ReduceMax(tmpReduce, scoreLocal, tmpReduce, this->tileKv);
            AscendC::PipeBarrier<PIPE_ALL>();
            float localMax = tmpReduce.GetValue(0);
            tmpBuf1.FreeTensor(tmpReduce);

            float newMax = (localMax > globalMax) ? localMax : globalMax;

            // Compute exp(scores - newMax)
            AscendC::Adds(scoreLocal, scoreLocal, -newMax, this->tileKv);
            AscendC::Exp(scoreLocal, scoreLocal, this->tileKv);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Sum of exp scores
            AscendC::LocalTensor<float> tmpReduce2 = tmpBuf1.AllocTensor<float>();
            AscendC::ReduceSum(tmpReduce2, scoreLocal, tmpReduce2, this->tileKv);
            AscendC::PipeBarrier<PIPE_ALL>();
            float localSum = tmpReduce2.GetValue(0);
            tmpBuf1.FreeTensor(tmpReduce2);

            // Update running sum: globalSum = globalSum * exp(globalMax - newMax) + localSum
            float correction = expf(globalMax - newMax);
            // Correct existing output accumulator
            if (kvt > 0) {
                AscendC::Muls(outLocal, outLocal, correction, this->dModel);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            globalSum = globalSum * correction + localSum;
            globalMax = newMax;

            // Accumulate weighted V for each d-tile
            for (uint32_t dt = 0; dt < this->dTileNum; dt++) {
                uint32_t dBase = dt * this->tileD;

                for (uint32_t ki = 0; ki < this->tileKv; ki++) {
                    float w = scoreLocal.GetValue(ki);

                    AscendC::LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
                    AscendC::DataCopy(vLocal, vGm[(kvBase + ki) * this->dModel + dBase], this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // outLocal[dBase:dBase+tileD] += w * vLocal
                    AscendC::Muls(vLocal, vLocal, w, this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    // We need to add to a sub-region of outLocal
                    // Use a temporary to extract, add, and put back
                    // Actually outLocal is contiguous - we can offset
                    // AscendC supports tensor indexing with []
                    AscendC::Add(outLocal[dBase], outLocal[dBase], vLocal, this->tileD);
                    AscendC::PipeBarrier<PIPE_ALL>();

                    inQueueV.FreeTensor(vLocal);
                }
            }

            outQueueScore.FreeTensor(scoreLocal);
        }

        // Normalize: outLocal /= globalSum
        if (globalSum > 0.0f) {
            float invSum = 1.0f / globalSum;
            AscendC::Muls(outLocal, outLocal, invSum, this->dModel);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // Copy output to global memory
        AscendC::DataCopy(outGm[0], outLocal, this->dModel);
        AscendC::PipeBarrier<PIPE_ALL>();

        outQueueOut.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueQ, inQueueK, inQueueV;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueScore, outQueueOut;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpBuf1, tmpBuf2;

    AscendC::GlobalTensor<float> qGm, kGm, vGm, outGm;

    uint32_t batchSize, qLen, kvLen, dModel;
    uint32_t kvTileNum, dTileNum;
    uint32_t tileKv, tileD;
};

extern "C" __global__ __aicore__ void kv_cached_chat_batch_attention_custom(
    GM_ADDR q, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelKvCachedAttn op;
    op.Init(q, k_cache, v_cache, output, workspace,
            tiling_data.batchSize, tiling_data.qLen, tiling_data.kvLen, tiling_data.dModel,
            tiling_data.kvTileNum, tiling_data.dTileNum);
    op.Process();
}
