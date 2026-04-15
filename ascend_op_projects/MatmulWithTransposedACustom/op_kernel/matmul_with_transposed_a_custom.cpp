
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmulTransA {
public:
    __aicore__ inline KernelMatmulTransA() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 uint32_t M, uint32_t K, uint32_t N)
    {
        this->M = M;
        this->K = K;
        this->N = N;
        
        // A is stored as (K, M), B as (K, N), C as (M, N)
        // C = A^T @ B, i.e. C[i][j] = sum_k A[k][i] * B[k][j]
        // We distribute rows of C (M dimension) across blocks
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->mPerBlock = (M + blockNum - 1) / blockNum;
        this->mStart = blockIdx * this->mPerBlock;
        if (this->mStart + this->mPerBlock > M) {
            this->mPerBlock = (this->mStart < M) ? (M - this->mStart) : 0;
        }
        
        aGm.SetGlobalBuffer((__gm__ float *)a, K * M);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);
        
        // We'll process tile by tile
        // tile size for K dimension
        this->kTileSize = 64;
        // tile size for N dimension
        this->nTileSize = 64;
        
        // Allocate buffers
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->kTileSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->kTileSize * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->nTileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->nTileSize * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (this->mPerBlock == 0) return;
        
        // For each row i of C that this block owns
        for (uint32_t mi = 0; mi < this->mPerBlock; mi++) {
            uint32_t i = this->mStart + mi;
            
            // Process N columns in tiles
            for (uint32_t nStart = 0; nStart < N; nStart += this->nTileSize) {
                uint32_t nLen = this->nTileSize;
                if (nStart + nLen > N) nLen = N - nStart;
                // Align nLen up to 8 for vector ops
                uint32_t nLenAligned = (nLen + 7) / 8 * 8;
                
                // Initialize accumulator to 0
                AscendC::LocalTensor<float> accLocal = outQueueC.AllocTensor<float>();
                AscendC::Duplicate(accLocal, (float)0.0f, nLenAligned);
                
                // Process K in tiles
                for (uint32_t kStart = 0; kStart < K; kStart += this->kTileSize) {
                    uint32_t kLen = this->kTileSize;
                    if (kStart + kLen > K) kLen = K - kStart;
                    
                    // For each k in this tile, accumulate A[k][i] * B[k][nStart:nStart+nLen]
                    for (uint32_t ki = 0; ki < kLen; ki++) {
                        uint32_t k = kStart + ki;
                        
                        // Load A[k][i] - single value
                        float aVal = 0.0f;
                        // Use DataCopy to load from global memory
                        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                        // A is (K, M), element A[k][i] is at offset k*M + i
                        // We need to copy at least 8 elements for alignment
                        uint32_t aOffset = k * M + (i / 8) * 8;
                        uint32_t aInnerOffset = i % 8;
                        uint32_t aCopyLen = 8;
                        AscendC::DataCopy(aLocal, aGm[aOffset], aCopyLen);
                        inQueueA.EnQue(aLocal);
                        AscendC::LocalTensor<float> aLocalDq = inQueueA.DeQue<float>();
                        aVal = aLocalDq.GetValue(aInnerOffset);
                        inQueueA.FreeTensor(aLocalDq);
                        
                        // Load B[k][nStart:nStart+nLen]
                        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                        uint32_t bOffset = k * N + nStart;
                        AscendC::DataCopy(bLocal, bGm[bOffset], nLenAligned);
                        inQueueB.EnQue(bLocal);
                        AscendC::LocalTensor<float> bLocalDq = inQueueB.DeQue<float>();
                        
                        // acc += aVal * bLocal
                        AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                        AscendC::Muls(tmpLocal, bLocalDq, aVal, nLenAligned);
                        AscendC::Add(accLocal, accLocal, tmpLocal, nLenAligned);
                        tmpBuf.FreeTensor(tmpLocal);
                        inQueueB.FreeTensor(bLocalDq);
                    }
                }
                
                // Store C[i][nStart:nStart+nLen]
                uint32_t cOffset = i * N + nStart;
                outQueueC.EnQue(accLocal);
                AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
                AscendC::DataCopy(cGm[cOffset], cLocal, nLenAligned);
                outQueueC.FreeTensor(cLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t M, K, N;
    uint32_t mPerBlock;
    uint32_t mStart;
    uint32_t kTileSize;
    uint32_t nTileSize;
};

extern "C" __global__ __aicore__ void matmul_with_transposed_a_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulTransA op;
    op.Init(a, b, c, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
