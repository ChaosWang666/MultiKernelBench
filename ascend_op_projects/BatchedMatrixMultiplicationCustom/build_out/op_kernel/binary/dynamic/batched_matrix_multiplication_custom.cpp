
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelBatchedMatMul {
public:
    __aicore__ inline KernelBatchedMatMul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 uint32_t batchSize, uint32_t M, uint32_t K, uint32_t N)
    {
        this->batchSize = batchSize;
        this->M = M;
        this->K = K;
        this->N = N;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // Each block handles a range of batches
        this->batchStart = blockIdx * batchSize / blockNum;
        uint32_t batchEnd = (blockIdx + 1) * batchSize / blockNum;
        this->batchCount = batchEnd - this->batchStart;

        uint32_t aMatSize = M * K;
        uint32_t bMatSize = K * N;
        uint32_t cMatSize = M * N;

        aGm.SetGlobalBuffer((__gm__ float *)a + this->batchStart * aMatSize, this->batchCount * aMatSize);
        bGm.SetGlobalBuffer((__gm__ float *)b + this->batchStart * bMatSize, this->batchCount * bMatSize);
        cGm.SetGlobalBuffer((__gm__ float *)c + this->batchStart * cMatSize, this->batchCount * cMatSize);

        // Determine tile size for rows of A / rows of C
        // We tile along M dimension. Each tile processes tileM rows.
        // For each tile: we need tileM * K for A tile, K * N for B (full), tileM * N for C tile
        // UB size is limited, so we also tile along N if needed.
        // Simple approach: tile along M with tileM rows, and along N with tileN cols
        // For each (tileM, tileN) block: need tileM*K + K*tileN + tileM*tileN floats in UB

        // Use conservative tile sizes
        this->tileM = 16;
        this->tileN = 16;
        this->tileK = K; // try to keep full K in one pass

        // If K is too large, we need to tile K as well
        // UB ~ 192KB for safe usage, let's use ~160KB = 40960 floats
        uint32_t maxFloats = 40960;
        
        // Adjust tileK if needed
        while (this->tileM * this->tileK + this->tileK * this->tileN + this->tileM * this->tileN > maxFloats && this->tileK > 16) {
            this->tileK /= 2;
        }
        // Further adjust tileM and tileN if still too large
        while (this->tileM * this->tileK + this->tileK * this->tileN + this->tileM * this->tileN > maxFloats && this->tileM > 1) {
            this->tileM /= 2;
        }
        while (this->tileM * this->tileK + this->tileK * this->tileN + this->tileM * this->tileN > maxFloats && this->tileN > 1) {
            this->tileN /= 2;
        }

        // Align sizes to 8 for float (32 bytes = 8 floats alignment)
        if (this->tileM < 8) this->tileM = 8;
        if (this->tileN < 8) this->tileN = 8;
        if (this->tileK < 8) this->tileK = 8;

        uint32_t aBufSize = this->tileM * this->tileK * sizeof(float);
        uint32_t bBufSize = this->tileK * this->tileN * sizeof(float);
        uint32_t cBufSize = this->tileM * this->tileN * sizeof(float);

        pipe.InitBuffer(inQueueA, BUFFER_NUM, aBufSize);
        pipe.InitBuffer(inQueueB, BUFFER_NUM, bBufSize);
        pipe.InitBuffer(outQueueC, BUFFER_NUM, cBufSize);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < this->batchCount; batch++) {
            uint32_t aBase = batch * M * K;
            uint32_t bBase = batch * K * N;
            uint32_t cBase = batch * M * N;
            
            for (uint32_t mStart = 0; mStart < M; mStart += tileM) {
                uint32_t curTileM = tileM;
                if (mStart + curTileM > M) curTileM = M - mStart;

                for (uint32_t nStart = 0; nStart < N; nStart += tileN) {
                    uint32_t curTileN = tileN;
                    if (nStart + curTileN > N) curTileN = N - nStart;

                    // Initialize output tile to zero
                    AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                    AscendC::Duplicate(cLocal, (float)0, curTileM * curTileN);

                    for (uint32_t kStart = 0; kStart < K; kStart += tileK) {
                        uint32_t curTileK = tileK;
                        if (kStart + curTileK > K) curTileK = K - kStart;

                        // Load A tile [curTileM x curTileK] from row-major A[mStart:mStart+curTileM, kStart:kStart+curTileK]
                        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                        for (uint32_t i = 0; i < curTileM; i++) {
                            uint32_t srcOffset = aBase + (mStart + i) * K + kStart;
                            AscendC::DataCopy(aLocal[i * curTileK], aGm[srcOffset], curTileK);
                        }
                        inQueueA.EnQue(aLocal);

                        // Load B tile [curTileK x curTileN] from row-major B[kStart:kStart+curTileK, nStart:nStart+curTileN]
                        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                        for (uint32_t i = 0; i < curTileK; i++) {
                            uint32_t srcOffset = bBase + (kStart + i) * N + nStart;
                            AscendC::DataCopy(bLocal[i * curTileN], bGm[srcOffset], curTileN);
                        }
                        inQueueB.EnQue(bLocal);

                        aLocal = inQueueA.DeQue<float>();
                        bLocal = inQueueB.DeQue<float>();

                        // Compute matmul: C[i][j] += sum_p A[i][p] * B[p][j]
                        for (uint32_t i = 0; i < curTileM; i++) {
                            for (uint32_t p = 0; p < curTileK; p++) {
                                // cLocal[i*curTileN .. i*curTileN+curTileN-1] += aLocal[i*curTileK+p] * bLocal[p*curTileN .. p*curTileN+curTileN-1]
                                AscendC::Axpy(cLocal[i * curTileN], bLocal[p * curTileN], aLocal[i * curTileK + p], curTileN);
                            }
                        }

                        inQueueA.FreeTensor(aLocal);
                        inQueueB.FreeTensor(bLocal);
                    }

                    // Store C tile
                    outQueueC.EnQue(cLocal);
                    cLocal = outQueueC.DeQue<float>();
                    for (uint32_t i = 0; i < curTileM; i++) {
                        uint32_t dstOffset = cBase + (mStart + i) * N + nStart;
                        AscendC::DataCopy(cGm[dstOffset], cLocal[i * curTileN], curTileN);
                    }
                    outQueueC.FreeTensor(cLocal);
                }
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t batchSize, M, K, N;
    uint32_t batchStart, batchCount;
    uint32_t tileM, tileN, tileK;
};

extern "C" __global__ __aicore__ void batched_matrix_multiplication_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelBatchedMatMul op;
    op.Init(a, b, c, tiling_data.batchSize, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
