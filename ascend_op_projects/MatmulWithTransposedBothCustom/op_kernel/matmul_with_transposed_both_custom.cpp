
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

// Each block computes a tile of output rows.
// A is (K, M) stored row-major, so A[k][m] = A[k*M + m]. A^T[m][k] = A[k*M + m].
// B is (N, K) stored row-major, so B[n][k] = B[n*K + k]. B^T[k][n] = B[n*K + k].
// C = A^T * B^T has shape (M, N). C[m][n] = sum_k A^T[m][k] * B^T[k][n] = sum_k A[k*M+m] * B[n*K+k]

class KernelMatmulTransposedBoth {
public:
    __aicore__ inline KernelMatmulTransposedBoth() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t K, uint32_t N)
    {
        this->M = M;
        this->K = K;
        this->N = N;
        
        aGm.SetGlobalBuffer((__gm__ float *)a, K * M);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * K);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);
        
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Distribute output rows (M rows) among blocks
        this->rowsPerBlock = (M + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > M) this->endRow = M;
        
        // Tile size for K dimension processing
        this->tileK = 256;
        if (this->tileK > K) this->tileK = K;
        
        // Tile size for N dimension
        this->tileN = 256;
        if (this->tileN > N) this->tileN = N;
        
        // Allocate buffers for tiles
        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileK * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileN * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->tileK * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (startRow >= endRow) return;
        
        for (uint32_t m = startRow; m < endRow; m++) {
            // Process row m of output: C[m][0..N-1]
            for (uint32_t nStart = 0; nStart < N; nStart += tileN) {
                uint32_t curTileN = tileN;
                if (nStart + curTileN > N) curTileN = N - nStart;
                
                // Allocate output tile and zero it
                AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                // Zero out the tile - use Duplicate to set all to 0
                AscendC::Duplicate(cLocal, (float)0.0f, this->tileN);
                
                for (uint32_t kStart = 0; kStart < K; kStart += tileK) {
                    uint32_t curTileK = tileK;
                    if (kStart + curTileK > K) curTileK = K - kStart;
                    
                    // For each n in [nStart, nStart+curTileN)
                    for (uint32_t ni = 0; ni < curTileN; ni++) {
                        uint32_t n = nStart + ni;
                        
                        // Load A^T[m][kStart..kStart+curTileK-1] = A[kStart*M+m], A[(kStart+1)*M+m], ...
                        // These are strided in global memory with stride M
                        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                        // Gather from A: elements at positions kStart*M+m, (kStart+1)*M+m, ...
                        // Since these are strided, we load them one by one
                        // For efficiency, use scalar accumulation
                        
                        // Load B^T[kStart..kStart+curTileK-1][n] = B[n*K+kStart..n*K+kStart+curTileK-1]
                        // These are contiguous
                        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                        AscendC::DataCopy(bLocal, bGm[n * K + kStart], (curTileK + 7) / 8 * 8);
                        
                        // Compute dot product via scalar
                        float sum = 0.0f;
                        for (uint32_t ki = 0; ki < curTileK; ki++) {
                            float aVal = *((__gm__ float*)(aGm.GetPhyAddr()) + (uint64_t)(kStart + ki) * M + m);
                            sum += aVal * bLocal.GetValue(ki);
                        }
                        
                        float prev = cLocal.GetValue(ni);
                        cLocal.SetValue(ni, prev + sum);
                        
                        inQueueA.FreeTensor(aLocal);
                        inQueueB.FreeTensor(bLocal);
                    }
                }
                
                // Write output tile
                // Pad to 8-element boundary for DataCopy
                uint32_t copyLen = (curTileN + 7) / 8 * 8;
                AscendC::DataCopy(cGm[m * N + nStart], cLocal, copyLen);
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
    uint32_t rowsPerBlock, startRow, endRow;
    uint32_t tileK, tileN;
};

extern "C" __global__ __aicore__ void matmul_with_transposed_both_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulTransposedBoth op;
    op.Init(a, b, c, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
