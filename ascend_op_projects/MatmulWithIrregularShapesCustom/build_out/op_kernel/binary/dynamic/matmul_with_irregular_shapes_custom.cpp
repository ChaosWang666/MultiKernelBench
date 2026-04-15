
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

// Tile size for K dimension processing
constexpr int32_t TILE_K = 64;
// Tile size for N dimension processing
constexpr int32_t TILE_N = 64;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t K, uint32_t N)
    {
        this->M = M;
        this->K = K;
        this->N = N;

        aGm.SetGlobalBuffer((__gm__ float *)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);

        // We process one row of A at a time, tile K and N
        // Buffer for a tile of A row: TILE_K elements
        uint32_t aAlignedSize = ((TILE_K + 7) / 8) * 8;
        // Buffer for a tile of B: TILE_K x TILE_N, but we process element by element
        // We'll use a simpler approach: load a chunk of A row (TILE_K), load corresponding B rows chunk (TILE_K x TILE_N)
        // and accumulate.
        // Actually for simplicity, let's do a straightforward approach:
        // For each row i of C, for each tile of N, accumulate over K.
        
        uint32_t bTileSize = ((TILE_K * TILE_N + 7) / 8) * 8;
        uint32_t nAlignedSize = ((TILE_N + 7) / 8) * 8;

        pipe.InitBuffer(inQueueA, BUFFER_NUM, aAlignedSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, bTileSize * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, nAlignedSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, nAlignedSize * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < M; i++) {
            // For each tile of N
            for (uint32_t nStart = 0; nStart < N; nStart += TILE_N) {
                uint32_t curTileN = (N - nStart < TILE_N) ? (N - nStart) : TILE_N;
                uint32_t nAligned = ((curTileN + 7) / 8) * 8;
                
                // Initialize accumulator to zero
                AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                AscendC::Duplicate(cLocal, (float)0.0f, nAligned);

                // Accumulate over K in tiles
                for (uint32_t kStart = 0; kStart < K; kStart += TILE_K) {
                    uint32_t curTileK = (K - kStart < TILE_K) ? (K - kStart) : TILE_K;
                    
                    // Load A[i, kStart:kStart+curTileK]
                    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                    uint32_t kAligned = ((curTileK + 7) / 8) * 8;
                    // Zero out first to handle padding
                    AscendC::Duplicate(aLocal, (float)0.0f, kAligned);
                    // Copy A elements
                    for (uint32_t kk = 0; kk < curTileK; kk++) {
                        aLocal.SetValue(kk, aGm.GetValue(i * K + kStart + kk));
                    }

                    // For each k in curTileK, accumulate A[i,k]*B[k, nStart:nStart+curTileN]
                    AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                    AscendC::LocalTensor<float> tLocal = tmpBuf.AllocTensor<float>();
                    
                    for (uint32_t kk = 0; kk < curTileK; kk++) {
                        float aVal = aLocal.GetValue(kk);
                        uint32_t bRowStart = (kStart + kk) * N + nStart;
                        
                        // Load B[kStart+kk, nStart:nStart+curTileN]
                        AscendC::Duplicate(tLocal, (float)0.0f, nAligned);
                        for (uint32_t nn = 0; nn < curTileN; nn++) {
                            tLocal.SetValue(nn, bGm.GetValue(bRowStart + nn));
                        }
                        
                        // tLocal = aVal * tLocal
                        AscendC::Muls(bLocal, tLocal, aVal, nAligned);
                        // cLocal += bLocal
                        AscendC::Add(cLocal, cLocal, bLocal, nAligned);
                    }
                    
                    tmpBuf.FreeTensor(tLocal);
                    inQueueB.FreeTensor(bLocal);
                    inQueueA.FreeTensor(aLocal);
                }
                
                // Store result C[i, nStart:nStart+curTileN]
                for (uint32_t nn = 0; nn < curTileN; nn++) {
                    cGm.SetValue(i * N + nStart + nn, cLocal.GetValue(nn));
                }
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
    uint32_t M;
    uint32_t K;
    uint32_t N;
};

extern "C" __global__ __aicore__ void matmul_with_irregular_shapes_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
